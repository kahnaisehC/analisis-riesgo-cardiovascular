#!/usr/bin/env python3
"""Build a styled A4 PDF from the authorized informe.md.

The source Markdown is deliberately left untouched.  Pandoc creates the
document body, then this script applies a small, self-contained DOCX design
system before LibreOffice renders the final PDF.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "pdf_build"
SOURCE = ROOT / "informe.md"
BODY_MD = BUILD / "informe_body.md"
SEED_MD = BUILD / "reference_seed.md"
REFERENCE_DOCX = BUILD / "reference.docx"
BODY_DOCX = BUILD / "informe_styled.docx"
OUT_PDF = ROOT / "informe.pdf"
TMP_PDF = BUILD / "informe_styled.pdf"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"

NS = {"w": W_NS, "r": R_NS, "wp": WP_NS}


def qn(prefix: str, local: str) -> str:
    return f"{{{ {'w': W_NS, 'r': R_NS, 'wp': WP_NS}[prefix] }}}{local}"


def w(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def r(local: str) -> str:
    return f"{{{R_NS}}}{local}"


def parse_xml(data: bytes) -> etree._Element:
    return etree.fromstring(data)


def xml_bytes(root: etree._Element) -> bytes:
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def child(parent: etree._Element, tag: str, **attrs: str) -> etree._Element:
    node = etree.SubElement(parent, w(tag))
    for key, value in attrs.items():
        node.set(w(key), value)
    return node


def ensure(parent: etree._Element, tag: str) -> etree._Element:
    node = parent.find(w(tag))
    return node if node is not None else child(parent, tag)


def replace_child(parent: etree._Element, tag: str) -> etree._Element:
    node = parent.find(w(tag))
    if node is not None:
        parent.remove(node)
    return child(parent, tag)


def set_attr_node(parent: etree._Element, tag: str, attr: str, value: str) -> etree._Element:
    node = parent.find(w(tag))
    if node is None:
        node = child(parent, tag)
    node.set(w(attr), value)
    return node


def set_font(rpr: etree._Element, family: str, size_half_points: int,
             color: str | None = None, bold: bool | None = None,
             italic: bool | None = None) -> None:
    fonts = rpr.find(w("rFonts"))
    if fonts is None:
        fonts = etree.Element(w("rFonts"))
        rpr.insert(0, fonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(w(attr), family)
    sz = rpr.find(w("sz"))
    if sz is None:
        sz = etree.SubElement(rpr, w("sz"))
    sz.set(w("val"), str(size_half_points))
    szcs = rpr.find(w("szCs"))
    if szcs is None:
        szcs = etree.SubElement(rpr, w("szCs"))
    szcs.set(w("val"), str(size_half_points))
    if color is not None:
        color_node = rpr.find(w("color"))
        if color_node is None:
            color_node = etree.SubElement(rpr, w("color"))
        color_node.set(w("val"), color)
        color_node.attrib.pop(w("themeColor"), None)
        color_node.attrib.pop(w("themeShade"), None)
    if bold is not None:
        bold_node = rpr.find(w("b"))
        if bold_node is None:
            bold_node = etree.SubElement(rpr, w("b"))
        if not bold:
            bold_node.set(w("val"), "0")
        else:
            bold_node.attrib.pop(w("val"), None)
    if italic is not None:
        italic_node = rpr.find(w("i"))
        if italic_node is None:
            italic_node = etree.SubElement(rpr, w("i"))
        if not italic:
            italic_node.set(w("val"), "0")
        else:
            italic_node.attrib.pop(w("val"), None)


def style_by_id(styles: etree._Element, style_id: str) -> etree._Element:
    node = styles.find(f".//{w('style')}[@{w('styleId')}='{style_id}']")
    if node is None:
        node = etree.SubElement(styles, w("style"))
        node.set(w("styleId"), style_id)
        node.set(w("type"), "paragraph")
        name = etree.SubElement(node, w("name"))
        name.set(w("val"), style_id)
    return node


def style_rpr(style: etree._Element) -> etree._Element:
    node = style.find(w("rPr"))
    return node if node is not None else etree.SubElement(style, w("rPr"))


def style_ppr(style: etree._Element) -> etree._Element:
    node = style.find(w("pPr"))
    return node if node is not None else etree.SubElement(style, w("pPr"))


def set_spacing(ppr: etree._Element, *, before: int, after: int,
                line: int = 276, rule: str = "auto") -> None:
    spacing = ppr.find(w("spacing"))
    if spacing is None:
        spacing = etree.SubElement(ppr, w("spacing"))
    spacing.set(w("before"), str(before))
    spacing.set(w("after"), str(after))
    spacing.set(w("line"), str(line))
    spacing.set(w("lineRule"), rule)


def set_style_defaults(styles: etree._Element) -> None:
    """Apply the report's typography and hierarchy."""
    defaults = styles.find(w("docDefaults"))
    if defaults is None:
        defaults = etree.Element(w("docDefaults"))
        styles.insert(0, defaults)
    rpr_default = defaults.find(w("rPrDefault"))
    if rpr_default is None:
        rpr_default = etree.SubElement(defaults, w("rPrDefault"))
    default_rpr = rpr_default.find(w("rPr"))
    if default_rpr is None:
        default_rpr = etree.SubElement(rpr_default, w("rPr"))
    set_font(default_rpr, "Noto Serif", 21, "263238")
    ppr_default = defaults.find(w("pPrDefault"))
    if ppr_default is None:
        ppr_default = etree.SubElement(defaults, w("pPrDefault"))
    default_ppr = ppr_default.find(w("pPr"))
    if default_ppr is None:
        default_ppr = etree.SubElement(ppr_default, w("pPr"))
    set_spacing(default_ppr, before=0, after=130, line=276)

    for sid in ("Normal", "BodyText", "FirstParagraph"):
        style = style_by_id(styles, sid)
        set_font(style_rpr(style), "Noto Serif", 21, "263238")
        set_spacing(style_ppr(style), before=0, after=130, line=276)

    compact = style_by_id(styles, "Compact")
    set_font(style_rpr(compact), "Noto Sans", 16, "263238")
    set_spacing(style_ppr(compact), before=0, after=15, line=210)

    h1 = style_by_id(styles, "Heading1")
    hppr = style_ppr(h1)
    set_spacing(hppr, before=0, after=180, line=276)
    for tag in ("keepNext", "keepLines"):
        if hppr.find(w(tag)) is None:
            etree.SubElement(hppr, w(tag))
    # Top-level sections begin on a fresh page after the index.
    if hppr.find(w("pageBreakBefore")) is None:
        etree.SubElement(hppr, w("pageBreakBefore"))
    outline = hppr.find(w("outlineLvl"))
    if outline is None:
        outline = etree.SubElement(hppr, w("outlineLvl"))
    outline.set(w("val"), "0")
    borders = hppr.find(w("pBdr"))
    if borders is None:
        borders = etree.SubElement(hppr, w("pBdr"))
    bottom = borders.find(w("bottom"))
    if bottom is None:
        bottom = etree.SubElement(borders, w("bottom"))
    bottom.attrib.update({
        w("val"): "single", w("sz"): "8", w("space"): "7", w("color"): "6FA3BE"
    })
    set_font(style_rpr(h1), "Noto Sans", 34, "17324D", bold=True)

    h2 = style_by_id(styles, "Heading2")
    h2ppr = style_ppr(h2)
    set_spacing(h2ppr, before=340, after=110, line=260)
    for tag in ("keepNext", "keepLines"):
        if h2ppr.find(w(tag)) is None:
            etree.SubElement(h2ppr, w(tag))
    h2borders = h2ppr.find(w("pBdr"))
    if h2borders is None:
        h2borders = etree.SubElement(h2ppr, w("pBdr"))
    h2bottom = h2borders.find(w("bottom"))
    if h2bottom is None:
        h2bottom = etree.SubElement(h2borders, w("bottom"))
    h2bottom.attrib.update({
        w("val"): "single", w("sz"): "5", w("space"): "4", w("color"): "A8C5D3"
    })
    set_font(style_rpr(h2), "Noto Sans", 26, "2F6F8F", bold=True)

    h3 = style_by_id(styles, "Heading3")
    set_spacing(style_ppr(h3), before=230, after=70, line=250)
    for tag in ("keepNext", "keepLines"):
        if style_ppr(h3).find(w(tag)) is None:
            etree.SubElement(style_ppr(h3), w(tag))
    set_font(style_rpr(h3), "Noto Sans", 23, "3D7D86", bold=True)

    for sid in ("Title", "Subtitle", "Author", "Date"):
        style = style_by_id(styles, sid)
        set_font(style_rpr(style), "Noto Sans", 24, "17324D")

    caption = style_by_id(styles, "Caption")
    set_font(style_rpr(caption), "Noto Sans", 17, "5A7184", italic=True)
    set_spacing(style_ppr(caption), before=50, after=170, line=220)
    for sid in ("ImageCaption", "TableCaption"):
        style = style_by_id(styles, sid)
        set_font(style_rpr(style), "Noto Sans", 17, "5A7184", italic=True)

    block = style_by_id(styles, "BlockText")
    set_font(style_rpr(block), "Noto Serif", 20, "37474F")
    set_spacing(style_ppr(block), before=100, after=100, line=260)

    source = style_by_id(styles, "SourceCode")
    sourceppr = style_ppr(source)
    set_spacing(sourceppr, before=100, after=150, line=210)
    if sourceppr.find(w("keepLines")) is None:
        etree.SubElement(sourceppr, w("keepLines"))
    ind = sourceppr.find(w("ind"))
    if ind is None:
        ind = etree.SubElement(sourceppr, w("ind"))
    ind.attrib.update({w("left"): "180", w("right"): "180"})
    shd = sourceppr.find(w("shd"))
    if shd is None:
        shd = etree.SubElement(sourceppr, w("shd"))
    shd.set(w("fill"), "F1F5F7")
    shd.set(w("val"), "clear")
    p_bdr = sourceppr.find(w("pBdr"))
    if p_bdr is None:
        p_bdr = etree.SubElement(sourceppr, w("pBdr"))
    left = p_bdr.find(w("left"))
    if left is None:
        left = etree.SubElement(p_bdr, w("left"))
    left.attrib.update({w("val"): "single", w("sz"): "18", w("space"): "7", w("color"): "6FA3BE"})
    set_font(style_rpr(source), "Noto Sans Mono", 17, "263238")
    verbatim = style_by_id(styles, "VerbatimChar")
    set_font(style_rpr(verbatim), "Noto Sans Mono", 17, "263238")

    hyperlink = style_by_id(styles, "Hyperlink")
    set_font(style_rpr(hyperlink), "Noto Sans", 21, "1769AA")


def decorate_table_style(styles: etree._Element) -> None:
    table = style_by_id(styles, "Table")
    tblpr = table.find(w("tblPr"))
    if tblpr is None:
        tblpr = etree.SubElement(table, w("tblPr"))
    margins = tblpr.find(w("tblCellMar"))
    if margins is None:
        margins = etree.SubElement(tblpr, w("tblCellMar"))
    for side, val in (("top", "45"), ("left", "75"), ("bottom", "45"), ("right", "75")):
        node = margins.find(w(side))
        if node is None:
            node = etree.SubElement(margins, w(side))
        node.attrib.update({w("type"): "dxa", w("w"): val})
    first = table.find(f"{w('tblStylePr')}[@{w('type')}='firstRow']")
    if first is None:
        first = etree.SubElement(table, w("tblStylePr"))
        first.set(w("type"), "firstRow")
    frpr = first.find(w("rPr"))
    if frpr is None:
        frpr = etree.SubElement(first, w("rPr"))
    set_font(frpr, "Noto Sans", 16, "FFFFFF", bold=True)
    ftcpr = first.find(w("tcPr"))
    if ftcpr is None:
        ftcpr = etree.SubElement(first, w("tcPr"))
    fshd = ftcpr.find(w("shd"))
    if fshd is None:
        fshd = etree.SubElement(ftcpr, w("shd"))
    fshd.attrib.update({w("val"): "clear", w("fill"): "17324D"})
    band = table.find(f"{w('tblStylePr')}[@{w('type')}='band1Horz']")
    if band is None:
        band = etree.SubElement(table, w("tblStylePr"))
        band.set(w("type"), "band1Horz")
    btcpr = band.find(w("tcPr"))
    if btcpr is None:
        btcpr = etree.SubElement(band, w("tcPr"))
    bshd = btcpr.find(w("shd"))
    if bshd is None:
        bshd = etree.SubElement(btcpr, w("shd"))
    bshd.attrib.update({w("val"): "clear", w("fill"): "F3F7F9"})


def set_ppr_border(ppr: etree._Element, side: str, color: str,
                   size: str = "6", space: str = "4") -> None:
    borders = ppr.find(w("pBdr"))
    if borders is None:
        borders = etree.SubElement(ppr, w("pBdr"))
    side_node = borders.find(w(side))
    if side_node is None:
        side_node = etree.SubElement(borders, w(side))
    side_node.attrib.update({
        w("val"): "single", w("sz"): size, w("space"): space, w("color"): color
    })


def run_element(text: str, *, family: str = "Noto Sans", size: int = 20,
                color: str = "263238", bold: bool = False,
                italic: bool = False, small_caps: bool = False) -> etree._Element:
    run = etree.Element(w("r"))
    rpr = etree.SubElement(run, w("rPr"))
    set_font(rpr, family, size, color, bold=bold, italic=italic)
    if small_caps:
        etree.SubElement(rpr, w("smallCaps"))
    node = etree.SubElement(run, w("t"))
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text
    return run


def paragraph_element(text: str = "", *, align: str = "left",
                      before: int = 0, after: int = 120,
                      family: str = "Noto Sans", size: int = 20,
                      color: str = "263238", bold: bool = False,
                      italic: bool = False, bottom_border: str | None = None,
                      style: str | None = None) -> etree._Element:
    p = etree.Element(w("p"))
    ppr = etree.SubElement(p, w("pPr"))
    if style:
        ps = etree.SubElement(ppr, w("pStyle"))
        ps.set(w("val"), style)
    jc = etree.SubElement(ppr, w("jc"))
    jc.set(w("val"), align)
    spacing = etree.SubElement(ppr, w("spacing"))
    spacing.attrib.update({w("before"): str(before), w("after"): str(after), w("line"): "240"})
    if bottom_border:
        set_ppr_border(ppr, "bottom", bottom_border, size="16", space="8")
    if text:
        p.append(run_element(text, family=family, size=size, color=color,
                             bold=bold, italic=italic))
    return p


def page_break_paragraph() -> etree._Element:
    p = etree.Element(w("p"))
    ppr = etree.SubElement(p, w("pPr"))
    spacing = etree.SubElement(ppr, w("spacing"))
    spacing.attrib.update({w("before"): "0", w("after"): "0"})
    run = etree.SubElement(p, w("r"))
    br = etree.SubElement(run, w("br"))
    br.set(w("type"), "page")
    return p


def cover_paragraphs() -> list[etree._Element]:
    # Twips: the large top spacing provides a calm, centered cover composition.
    return [
        paragraph_element(
            "UTN · FACULTAD REGIONAL RESISTENCIA",
            align="center", before=1550, after=420, size=17,
            color="5A7184", bold=True, family="Noto Sans",
        ),
        paragraph_element(
            "INTELIGENCIA ARTIFICIAL PARA DATOS",
            align="center", before=0, after=850, size=19,
            color="2F6F8F", bold=True, family="Noto Sans",
        ),
        paragraph_element(
            "Trabajo Práctico",
            align="center", before=0, after=140, size=38,
            color="17324D", bold=True, family="Noto Sans",
        ),
        paragraph_element(
            "Decision Tree vs Random Forest",
            align="center", before=0, after=280, size=30,
            color="2F6F8F", bold=True, family="Noto Sans",
            bottom_border="6FA3BE",
        ),
        paragraph_element(
            "Comparación de algoritmos de clasificación sobre el Framingham Heart Study",
            align="center", before=180, after=720, size=19,
            color="37474F", italic=True, family="Noto Serif",
        ),
        paragraph_element(
            "Dataset: data_cardiovascular_risk.csv",
            align="center", before=0, after=180, size=18,
            color="263238", family="Noto Sans",
        ),
        paragraph_element(
            "Integrantes",
            align="center", before=580, after=130, size=18,
            color="5A7184", bold=True, family="Noto Sans",
        ),
        paragraph_element(
            "Ian Hakanson · Joaquin Valdez · Lucas Rabinovich",
            align="center", before=0, after=650, size=20,
            color="263238", family="Noto Sans",
        ),
        paragraph_element(
            "Ciclo lectivo 2026",
            align="center", before=650, after=0, size=17,
            color="5A7184", family="Noto Sans",
        ),
        page_break_paragraph(),
    ]


def add_header_footer(package: dict[str, bytes], rels_root: etree._Element,
                      content_types: etree._Element, sectpr: etree._Element) -> None:
    rels = rels_root.findall(f"{{{REL_NS}}}Relationship")
    used = {node.get("Id") for node in rels}

    def next_rid() -> str:
        i = 1
        while f"rId{i}" in used:
            i += 1
        value = f"rId{i}"
        used.add(value)
        return value

    header_id, footer_id = next_rid(), next_rid()
    for rid, rel_type, target in (
        (header_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header", "header.xml"),
        (footer_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer", "footer.xml"),
    ):
        rel = etree.SubElement(rels_root, f"{{{REL_NS}}}Relationship")
        rel.attrib.update({"Id": rid, "Type": rel_type, "Target": target})

    for part, content_type in (
        ("header.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"),
        ("footer.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"),
    ):
        if not any(node.get("PartName") == f"/word/{part}" for node in content_types):
            override = etree.SubElement(content_types, f"{{{CT_NS}}}Override")
            override.attrib.update({"PartName": f"/word/{part}", "ContentType": content_type})

    # Header with a subtle rule and a two-sided running label.
    header = etree.Element(
        f"{{{W_NS}}}hdr",
        nsmap={"w": W_NS, "r": R_NS},
    )
    hp = etree.SubElement(header, w("p"))
    hppr = etree.SubElement(hp, w("pPr"))
    tabs = etree.SubElement(hppr, w("tabs"))
    tab = etree.SubElement(tabs, w("tab"))
    tab.attrib.update({w("val"): "right", w("pos"): "9400"})
    set_ppr_border(hppr, "bottom", "A8C5D3", size="6", space="5")
    hp.append(run_element("IAD · UTN F.R. RESISTENCIA", family="Noto Sans",
                          size=15, color="5A7184", bold=True))
    rtab = etree.SubElement(hp, w("r"))
    etree.SubElement(rtab, w("tab"))
    hp.append(run_element("INFORME ACADÉMICO · 2026", family="Noto Sans",
                          size=15, color="5A7184"))

    # Footer with a dynamic page number.  LibreOffice resolves PAGE on export.
    footer = etree.Element(
        f"{{{W_NS}}}ftr",
        nsmap={"w": W_NS, "r": R_NS},
    )
    fp = etree.SubElement(footer, w("p"))
    fppr = etree.SubElement(fp, w("pPr"))
    ftabs = etree.SubElement(fppr, w("tabs"))
    ftab = etree.SubElement(ftabs, w("tab"))
    ftab.attrib.update({w("val"): "right", w("pos"): "9400"})
    set_ppr_border(fppr, "top", "A8C5D3", size="6", space="5")
    fp.append(run_element("Decision Tree vs Random Forest", family="Noto Sans",
                          size=15, color="5A7184"))
    frtab = etree.SubElement(fp, w("r"))
    etree.SubElement(frtab, w("tab"))
    fp.append(run_element("PÁGINA ", family="Noto Sans", size=15, color="5A7184"))
    fld = etree.SubElement(fp, f"{{{W_NS}}}fldSimple")
    fld.set(w("instr"), "PAGE")
    fld.append(run_element("1", family="Noto Sans", size=15, color="5A7184"))

    package["word/header.xml"] = xml_bytes(header)
    package["word/footer.xml"] = xml_bytes(footer)

    # A title-page section has no running header/footer in LibreOffice when
    # titlePg is enabled; subsequent pages use the references below.
    for tag, rid, typ in (
        ("headerReference", header_id, "default"),
        ("footerReference", footer_id, "default"),
    ):
        node = sectpr.find(w(tag))
        if node is None:
            node = etree.Element(w(tag))
            sectpr.insert(0, node)
        node.attrib.update({r("id"): rid, w("type"): typ})
    if sectpr.find(w("titlePg")) is None:
        etree.SubElement(sectpr, w("titlePg"))


def add_section_properties(document: etree._Element, rels_root: etree._Element,
                           content_types: etree._Element, package: dict[str, bytes]) -> None:
    body = document.find(w("body"))
    assert body is not None
    sectpr = body.find(w("sectPr"))
    if sectpr is None:
        sectpr = etree.SubElement(body, w("sectPr"))
    # Remove references left by a previous build, if any.
    for node in list(sectpr):
        if node.tag in {w("headerReference"), w("footerReference"), w("titlePg"),
                        w("pgSz"), w("pgMar"), w("cols"), w("docGrid")}:
            sectpr.remove(node)
    add_header_footer(package, rels_root, content_types, sectpr)
    pg = etree.SubElement(sectpr, w("pgSz"))
    pg.attrib.update({w("w"): "11906", w("h"): "16838"})
    mar = etree.SubElement(sectpr, w("pgMar"))
    mar.attrib.update({
        w("top"): "1450", w("right"): "1260", w("bottom"): "1250",
        w("left"): "1260", w("header"): "650", w("footer"): "650", w("gutter"): "0",
    })
    cols = etree.SubElement(sectpr, w("cols"))
    cols.set(w("num"), "1")
    docgrid = etree.SubElement(sectpr, w("docGrid"))
    docgrid.attrib.update({w("linePitch"): "276"})


def insert_cover(document: etree._Element) -> None:
    body = document.find(w("body"))
    assert body is not None
    # The cover must precede the Markdown body.  Inserting before ``sectPr``
    # places it at the end of the document, which makes the generated cover
    # collide with the first page of the report.
    insertion_index = 0
    for node in reversed(cover_paragraphs()):
        body.insert(insertion_index, node)


def text_of(node: etree._Element) -> str:
    return "".join(node.itertext())


def set_cell_shading(cell: etree._Element, fill: str) -> None:
    tcpr = cell.find(w("tcPr"))
    if tcpr is None:
        tcpr = etree.Element(w("tcPr"))
        cell.insert(0, tcpr)
    shd = tcpr.find(w("shd"))
    if shd is None:
        shd = etree.SubElement(tcpr, w("shd"))
    shd.attrib.update({w("val"): "clear", w("fill"): fill})


def set_cell_width(cell: etree._Element, width: int) -> None:
    tcpr = cell.find(w("tcPr"))
    if tcpr is None:
        tcpr = etree.Element(w("tcPr"))
        cell.insert(0, tcpr)
    tcw = tcpr.find(w("tcW"))
    if tcw is None:
        tcw = etree.SubElement(tcpr, w("tcW"))
    tcw.attrib.update({w("type"): "dxa", w("w"): str(width)})


def set_run_size(run: etree._Element, size: int, *, color: str | None = None,
                 bold: bool | None = None) -> None:
    rpr = run.find(w("rPr"))
    if rpr is None:
        rpr = etree.Element(w("rPr"))
        run.insert(0, rpr)
    sz = rpr.find(w("sz"))
    if sz is None:
        sz = etree.SubElement(rpr, w("sz"))
    sz.set(w("val"), str(size))
    szcs = rpr.find(w("szCs"))
    if szcs is None:
        szcs = etree.SubElement(rpr, w("szCs"))
    szcs.set(w("val"), str(size))
    if color is not None:
        col = rpr.find(w("color"))
        if col is None:
            col = etree.SubElement(rpr, w("color"))
        col.attrib.update({w("val"): color})
        col.attrib.pop(w("themeColor"), None)
    if bold is not None:
        b = rpr.find(w("b"))
        if b is None:
            b = etree.SubElement(rpr, w("b"))
        if not bold:
            b.set(w("val"), "0")
        else:
            b.attrib.pop(w("val"), None)


def decorate_tables_and_images(document: etree._Element) -> None:
    body = document.find(w("body"))
    assert body is not None
    available_twips = 9386
    max_emu = available_twips * 635

    for table in body.findall(f".//{w('tbl')}"):
        rows = table.findall(w("tr"))
        if not rows:
            continue
        cells = rows[0].findall(w("tc"))
        ncols = len(cells)
        if ncols == 0:
            continue
        # Give each table a predictable width, then allocate space according
        # to the longest header.  Wide metric tables remain readable at 7.5 pt.
        header_lengths = [max(3, len(re.sub(r"\\s+", " ", text_of(c)).strip()))
                          for c in cells]
        weights = [max(480, min(2600, 500 + length * 62))
                   for length in header_lengths]
        scale = available_twips / sum(weights)
        widths = [max(480, int(round(value * scale))) for value in weights]
        # Correct rounding so the table exactly fits the text area.
        widths[-1] += available_twips - sum(widths)
        grid = table.find(w("tblGrid"))
        if grid is not None:
            gridcols = grid.findall(w("gridCol"))
            for i, gridcol in enumerate(gridcols[:ncols]):
                gridcol.set(w("w"), str(widths[i]))
        tblpr = table.find(w("tblPr"))
        if tblpr is None:
            tblpr = etree.Element(w("tblPr"))
            table.insert(0, tblpr)
        tblw = tblpr.find(w("tblW"))
        if tblw is None:
            tblw = etree.SubElement(tblpr, w("tblW"))
        tblw.attrib.update({w("type"): "dxa", w("w"): str(available_twips)})
        layout = tblpr.find(w("tblLayout"))
        if layout is None:
            layout = etree.SubElement(tblpr, w("tblLayout"))
        layout.set(w("type"), "fixed")
        jc = tblpr.find(w("jc"))
        if jc is None:
            jc = etree.SubElement(tblpr, w("jc"))
        jc.set(w("val"), "center")

        for row_index, row in enumerate(rows):
            if row.find(w("trPr")) is None:
                row.insert(0, etree.Element(w("trPr")))
            if row_index == 0:
                trpr = row.find(w("trPr"))
                if trpr.find(w("tblHeader")) is None:
                    etree.SubElement(trpr, w("tblHeader")).set(w("val"), "true")
            for col_index, cell in enumerate(row.findall(w("tc"))):
                if col_index < len(widths):
                    set_cell_width(cell, widths[col_index])
                # Alternate rows, keeping the palette restrained.
                if row_index > 0 and row_index % 2 == 0:
                    set_cell_shading(cell, "F3F7F9")
                elif row_index == 0:
                    set_cell_shading(cell, "17324D")
                tcpr = cell.find(w("tcPr"))
                if tcpr is None:
                    tcpr = etree.Element(w("tcPr"))
                    cell.insert(0, tcpr)
                borders = tcpr.find(w("tcBorders"))
                if borders is None:
                    borders = etree.SubElement(tcpr, w("tcBorders"))
                for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                    border = borders.find(w(side))
                    if border is None:
                        border = etree.SubElement(borders, w(side))
                    border.attrib.update({
                        w("val"): "single", w("sz"): "3", w("space"): "0",
                        w("color"): "C5D4DB",
                    })
                for p in cell.findall(w("p")):
                    ppr = p.find(w("pPr"))
                    if ppr is None:
                        ppr = etree.Element(w("pPr"))
                        p.insert(0, ppr)
                    spacing = ppr.find(w("spacing"))
                    if spacing is None:
                        spacing = etree.SubElement(ppr, w("spacing"))
                    spacing.attrib.update({w("before"): "0", w("after"): "10", w("line"): "190"})
                    if row_index == 0:
                        jc = ppr.find(w("jc"))
                        if jc is None:
                            jc = etree.SubElement(ppr, w("jc"))
                        jc.set(w("val"), "center")
                    for run in p.findall(f".//{w('r')}"):
                        set_run_size(run, 15 if ncols >= 9 else (16 if ncols >= 7 else 17),
                                     color="FFFFFF" if row_index == 0 else None,
                                     bold=True if row_index == 0 else None)

    # Fit every figure to the A4 text block and center its paragraph.
    for inline in body.findall(f".//{w('inline')}"):
        extent = inline.find(wp("extent")) if False else inline.find(f"{{{WP_NS}}}extent")
        if extent is None:
            continue
        cx = int(extent.get("cx", "0"))
        cy = int(extent.get("cy", "0"))
        if cx > max_emu and cx:
            new_cx = max_emu
            new_cy = max(1, int(round(cy * new_cx / cx)))
            extent.set("cx", str(new_cx))
            extent.set("cy", str(new_cy))
            aext = inline.find(f".//{{http://schemas.openxmlformats.org/drawingml/2006/main}}ext")
            if aext is not None:
                aext.set("cx", str(new_cx))
                aext.set("cy", str(new_cy))
        parent = inline
        while parent is not None and parent.tag != w("p"):
            parent = parent.getparent()
        if parent is not None:
            ppr = parent.find(w("pPr"))
            if ppr is None:
                ppr = etree.Element(w("pPr"))
                parent.insert(0, ppr)
            jc = ppr.find(w("jc"))
            if jc is None:
                jc = etree.SubElement(ppr, w("jc"))
            jc.set(w("val"), "center")


def wp(local: str) -> str:
    return f"{{{WP_NS}}}{local}"


def make_body_source() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    # The cover carries the report identity block from lines 1–9.  All
    # substantive Markdown, including the manual index and every figure, is
    # copied verbatim after the opening separator.
    if len(lines) < 11:
        raise RuntimeError("informe.md no tiene la estructura esperada")
    BODY_MD.write_text("".join(lines[10:]), encoding="utf-8")


def make_seed() -> None:
    SEED_MD.write_text(
        "# Seed\n\nBody.\n\n## Section\n\n| A | B |\n|---|---|\n| x | y |\n",
        encoding="utf-8",
    )


def create_reference_docx() -> None:
    if not REFERENCE_DOCX.exists():
        subprocess.run(
            ["pandoc", str(SEED_MD), "-o", str(REFERENCE_DOCX)],
            check=True,
            cwd=ROOT,
        )


def build_pandoc_docx() -> None:
    cmd = [
        "pandoc", str(BODY_MD), "-o", str(BODY_DOCX),
        "--from=markdown+implicit_figures",
        "--reference-doc", str(REFERENCE_DOCX),
        "--highlight-style=tango",
        "--resource-path", f"{ROOT}:{BUILD}:{ROOT / 'results' / 'figuras'}",
        "--metadata", "lang=es-AR",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def style_docx() -> None:
    with zipfile.ZipFile(BODY_DOCX, "r") as zin:
        package = {name: zin.read(name) for name in zin.namelist()}

    styles = parse_xml(package["word/styles.xml"])
    set_style_defaults(styles)
    decorate_table_style(styles)
    package["word/styles.xml"] = xml_bytes(styles)

    document = parse_xml(package["word/document.xml"])
    rels = parse_xml(package["word/_rels/document.xml.rels"])
    content_types = parse_xml(package["[Content_Types].xml"])
    insert_cover(document)
    add_section_properties(document, rels, content_types, package)
    decorate_tables_and_images(document)
    package["word/document.xml"] = xml_bytes(document)
    package["word/_rels/document.xml.rels"] = xml_bytes(rels)
    package["[Content_Types].xml"] = xml_bytes(content_types)

    # Update document metadata without touching the Markdown.
    if "docProps/core.xml" in package:
        core = parse_xml(package["docProps/core.xml"])
        dc = {"http://purl.org/dc/elements/1.1/": "dc"}
        title = core.find("{http://purl.org/dc/elements/1.1/}title")
        if title is None:
            title = etree.SubElement(core, "{http://purl.org/dc/elements/1.1/}title")
        title.text = "TP IAD 2026 — Decision Tree vs Random Forest"
        creator = core.find("{http://purl.org/dc/elements/1.1/}creator")
        if creator is None:
            creator = etree.SubElement(core, "{http://purl.org/dc/elements/1.1/}creator")
        creator.text = "Ian Hakanson · Joaquin Valdez · Lucas Rabinovich"
        package["docProps/core.xml"] = xml_bytes(core)

    if BODY_DOCX.exists():
        BODY_DOCX.unlink()
    with zipfile.ZipFile(BODY_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in package.items():
            zout.writestr(name, data)


def convert_pdf() -> None:
    if TMP_PDF.exists():
        TMP_PDF.unlink()
    subprocess.run(
        [
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", str(BUILD), str(BODY_DOCX),
        ],
        check=True, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    generated = BUILD / f"{BODY_DOCX.stem}.pdf"
    if not generated.exists() or generated.stat().st_size < 50_000:
        raise RuntimeError("LibreOffice no produjo un PDF válido")
    generated.replace(TMP_PDF)
    TMP_PDF.replace(OUT_PDF)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    make_body_source()
    make_seed()
    create_reference_docx()
    build_pandoc_docx()
    style_docx()
    convert_pdf()
    print(f"Generado: {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
