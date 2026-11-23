"""Build the standalone VisAge specification from its Markdown source.

Run with the bundled document Python runtime. This does not import model code,
download weights, train models, or modify the working application.
"""
from __future__ import annotations

from pathlib import Path
import re

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp/visage_documentation"
OUTPUT = ROOT / "output/docx/VisAge_Project_Documentation.docx"
SOURCE = ROOT / "docs/VisAge_Project_Documentation.md"
BLACK = RGBColor(0, 0, 0)


def diagram():
    canvas = Image.new("RGB", (2000, 1310), "white")
    draw = ImageDraw.Draw(canvas)
    normal = r"C:\Windows\Fonts\arial.ttf"
    bold = r"C:\Windows\Fonts\arialbd.ttf"

    def text(x, y, words, size=36, heavy=False):
        font = ImageFont.truetype(bold if heavy else normal, size)
        draw.text((x, y), words, font=font, fill="#172a40", anchor="mm", align="center")

    def box(x0, y0, x1, y1, title, detail, fill="#f5f8fc"):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=17, fill=fill, outline="#75869b", width=3)
        text((x0 + x1) / 2, y0 + 57, title, 42, True)
        text((x0 + x1) / 2, y0 + 116, detail, 34)

    def arrow(x0, y0, x1, y1):
        draw.line((x0, y0, x1, y1), fill="#42658d", width=5)
        if y1 > y0:
            draw.polygon([(x1, y1), (x1 - 13, y1 - 22), (x1 + 13, y1 - 22)], fill="#42658d")

    box(160, 20, 1840, 185, "Browser input", "Upload a photo or capture one still camera frame")
    arrow(1000, 185, 1000, 265)
    box(120, 265, 1880, 435, "FastAPI and shared face processing", "Validate   Decode   Orient   YuNet   Select   Align", "#edf3fa")
    draw.line((1000, 435, 1000, 480), fill="#42658d", width=5)
    draw.line((375, 480, 1625, 480), fill="#42658d", width=5)
    positions = [(105, 645, "ConvNeXt V2 Nano"), (730, 1270, "EfficientNetV2 S"), (1355, 1895, "Swin V2 Tiny")]
    for left, right, name in positions:
        center = (left + right) // 2
        arrow(center, 480, center, 555)
        box(left, 555, right, 760, name, "Saved model transform")
        text(center, 715, "Three age group logits", 31)
        draw.line((center, 760, center, 830), fill="#42658d", width=5)
    draw.line((375, 830, 1625, 830), fill="#42658d", width=5)
    arrow(1000, 830, 1000, 905)
    box(160, 905, 1840, 1075, "Browser estimates and comparison", "Selected face overlay   Age group cards   Processing times", "#edf3fa")
    text(1000, 1160, "Offline training produces verified checkpoints and evaluation records", 34)
    text(1000, 1220, "The dashboard reads aggregate evidence with matching artifact identities", 32)
    path = WORK / "architecture.png"
    canvas.save(path, dpi=(300, 300))
    return path


def color_font(style, size, bold=False, font="Calibri"):
    style.font.name = font
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = BLACK
    style.font.underline = False
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is not None:
        for key in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn("w:" + key), None)
        fonts.set(qn("w:ascii"), font); fonts.set(qn("w:hAnsi"), font)
    color = rpr.find(qn("w:color"))
    if color is not None:
        for key in ("themeColor", "themeTint", "themeShade"):
            color.attrib.pop(qn("w:" + key), None)


def hyperlink(paragraph, label, url):
    relation = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    node = OxmlElement("w:hyperlink"); node.set(qn("r:id"), relation)
    run = OxmlElement("w:r"); props = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), "184E7A"); props.append(color)
    underline = OxmlElement("w:u"); underline.set(qn("w:val"), "single"); props.append(underline)
    run.append(props); word = OxmlElement("w:t"); word.text = label; run.append(word); node.append(run); paragraph._p.append(node)


def inline(paragraph, text):
    pattern = r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\(https?://[^)]+\)|https?://\S+)"
    for token in re.split(pattern, text):
        if not token:
            continue
        link = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", token)
        if link:
            hyperlink(paragraph, link[1], link[2])
        elif token.startswith("http"):
            hyperlink(paragraph, token, token)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1]); run.font.name = "Consolas"; run.font.size = Pt(9)
        elif token.startswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        else:
            paragraph.add_run(token)


def add_field(paragraph, instruction):
    run = paragraph.add_run(); begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin"); run._r.append(begin)
    run = paragraph.add_run(); command = OxmlElement("w:instrText"); command.set(qn("xml:space"), "preserve"); command.text = instruction; run._r.append(command)
    run = paragraph.add_run(); separate = OxmlElement("w:fldChar"); separate.set(qn("w:fldCharType"), "separate"); run._r.append(separate)
    paragraph.add_run("1")
    run = paragraph.add_run(); end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end"); run._r.append(end)


def list_id(doc, numbered):
    root = doc.part.numbering_part.element
    abstract_ids = [int(item.get(qn("w:abstractNumId"))) for item in root.findall(qn("w:abstractNum"))]
    num_ids = [int(item.get(qn("w:numId"))) for item in root.findall(qn("w:num"))]
    aid, nid = max(abstract_ids, default=-1) + 1, max(num_ids, default=0) + 1
    abstract = OxmlElement("w:abstractNum"); abstract.set(qn("w:abstractNumId"), str(aid))
    level = OxmlElement("w:lvl"); level.set(qn("w:ilvl"), "0")
    for name, value in [("start", "1"), ("numFmt", "decimal" if numbered else "bullet"), ("lvlText", "%1." if numbered else "\u2022"), ("lvlJc", "left")]:
        node = OxmlElement("w:" + name); node.set(qn("w:val"), value); level.append(node)
    paragraph = OxmlElement("w:pPr"); indent = OxmlElement("w:ind"); indent.set(qn("w:left"), "300"); indent.set(qn("w:hanging"), "240"); paragraph.append(indent); level.append(paragraph)
    abstract.append(level)
    first_num = next((i for i, item in enumerate(root) if item.tag == qn("w:num")), len(root))
    root.insert(first_num, abstract)
    num = OxmlElement("w:num"); num.set(qn("w:numId"), str(nid)); reference = OxmlElement("w:abstractNumId"); reference.set(qn("w:val"), str(aid)); num.append(reference); root.append(num)
    return nid


def make_table(doc, rows, total):
    headers = rows[0]
    if len(headers) == 2:
        ratios = [0.29, 0.71]
    elif len(headers) == 4:
        ratios = [0.23, 0.28, 0.29, 0.20]
    elif headers[0] == "Phase":
        ratios = [0.22, 0.40, 0.38]
    elif headers[0] in {"ID", "Class ID"}:
        ratios = [0.11, 0.43, 0.46]
    elif headers[0] == "Endpoint":
        ratios = [0.27, 0.24, 0.49]
    elif headers[0] == "Deliverable":
        ratios = [0.24, 0.44, 0.32]
    else:
        ratios = [0.25, 0.35, 0.40]
    table = doc.add_table(rows=0, cols=len(headers)); table.alignment = WD_TABLE_ALIGNMENT.CENTER; table.autofit = False
    for column, ratio in zip(table.columns, ratios):
        column.width = Inches(total * ratio)
    props = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + edge); border.set(qn("w:val"), "single"); border.set(qn("w:sz"), "4"); border.set(qn("w:color"), "D9D9D9"); borders.append(border)
    props.append(borders)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        row_props = table.rows[-1]._tr.get_or_add_trPr()
        row_props.append(OxmlElement("w:cantSplit"))
        if i == 0:
            row_props.append(OxmlElement("w:tblHeader"))
        for j, (cell, words, ratio) in enumerate(zip(cells, row, ratios)):
            cell.width = Inches(total * ratio); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcprops = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for edge, val in [("top", "85"), ("bottom", "85"), ("left", "105"), ("right", "105")]:
                item = OxmlElement("w:" + edge); item.set(qn("w:w"), val); item.set(qn("w:type"), "dxa"); margins.append(item)
            tcprops.append(margins)
            shade = OxmlElement("w:shd"); shade.set(qn("w:fill"), "17365D" if i == 0 else "F1F5FA" if i % 2 == 0 else "FFFFFF"); tcprops.append(shade)
            p = cell.paragraphs[0]; p.style = doc.styles["Table Body"]
            p.paragraph_format.keep_with_next = i == 0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (j == 0 and headers[0] in {"ID", "Phase", "Class ID"}) else WD_ALIGN_PARAGRAPH.LEFT
            inline(p, words)
            if i == 0:
                for run in p.runs:
                    run.bold = True; run.font.color.rgb = RGBColor(255, 255, 255)
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2); p.paragraph_format.space_before = Pt(0); p.paragraph_format.line_spacing = Pt(3)


def main():
    WORK.mkdir(parents=True, exist_ok=True); OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = diagram(); doc = Document()
    doc.core_properties.title = "VisAge Project Documentation"
    doc.core_properties.subject = "Pretrained facial age group classification and web application development"
    doc.core_properties.author = "VisAge Project"
    doc.core_properties.keywords = "VisAge, computer vision, transfer learning, project specification"
    doc.core_properties.comments = ""
    section = doc.sections[0]
    section.page_width = Inches(8.2677); section.page_height = Inches(11.6929)
    section.top_margin = Inches(.75); section.bottom_margin = Inches(.75)
    section.left_margin = Inches(.8); section.right_margin = Inches(.8)
    section.header_distance = Inches(.3); section.footer_distance = Inches(.35)
    section.different_first_page_header_footer = True
    total = 8.2677 - 1.6
    normal = doc.styles["Normal"]; color_font(normal, 10.5)
    normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.widow_control = True
    for name, size, bold in [("Title", 31, True), ("Subtitle", 16, False), ("Heading 1", 18, True), ("Heading 2", 12, True), ("Heading 3", 11, True)]:
        style = doc.styles[name]; color_font(style, size, bold)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(13 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(7)
    doc.styles["Title"].paragraph_format.space_before = Pt(48)
    doc.styles["Subtitle"].paragraph_format.space_after = Pt(25)
    doc.styles["Heading 1"].paragraph_format.page_break_before = False
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(19)
    for name, size, face in [("Code Block", 8.5, "Consolas"), ("Table Body", 9.4, "Calibri"), ("Caption", 9.2, "Calibri")]:
        style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH) if name not in doc.styles else doc.styles[name]
        color_font(style, size, font=face); style.paragraph_format.line_spacing = 1.05; style.paragraph_format.space_after = Pt(4)
    doc.styles["Code Block"].paragraph_format.left_indent = Inches(.12)
    doc.styles["Caption"].font.italic = False
    for level in (1, 2):
        name = f"TOC {level}"
        style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH) if name not in doc.styles else doc.styles[name]
        color_font(style, 10 if level == 1 else 9, level == 1)
        style.paragraph_format.space_after = Pt(3); style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.left_indent = Inches(0 if level == 1 else .18)
    footer = section.footer.paragraphs[0]; footer.paragraph_format.tab_stops.clear_all(); footer.paragraph_format.tab_stops.add_tab_stop(Inches(total), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.SPACES)
    footer.add_run("VisAge Project Documentation\tPage "); add_field(footer, " PAGE "); footer.add_run(" of "); add_field(footer, " NUMPAGES ")
    for run in footer.runs:
        run.font.name = "Calibri"; run.font.size = Pt(8); run.font.color.rgb = BLACK
    settings = doc.settings.element; update = OxmlElement("w:updateFields"); update.set(qn("w:val"), "true"); settings.append(update)
    language = OxmlElement("w:themeFontLang"); language.set(qn("w:val"), "en-US"); settings.append(language)
    lines = SOURCE.read_text(encoding="utf-8").splitlines(); i = 0; first_h2 = True; code = False; current_list = None; code_paragraphs = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            current_list = None; i += 1; continue
        if line.startswith("```"):
            if code and code_paragraphs:
                code_paragraphs[-1].paragraph_format.keep_with_next = False
                code_paragraphs = []
            code = not code; i += 1; continue
        if code:
            p = doc.add_paragraph(style="Code Block"); p.add_run(lines[i]); p.paragraph_format.keep_with_next = True; code_paragraphs.append(p); i += 1; continue
        if line == "\\pagebreak":
            doc.add_page_break(); i += 1; continue
        if line == "\\toc":
            p = doc.add_paragraph(); add_field(p, ' TOC \\o "1-1" \\h \\z \\u '); i += 1; continue
        if line == "\\architecture":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(); run.add_picture(str(image), width=Inches(total))
            props = run._r.xpath(".//wp:docPr")[0]; props.set("descr", "Image input passes through shared face processing to three pretrained classifiers and browser estimates.")
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [word.strip() for word in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", word.replace(" ", "")) for word in row):
                    rows.append(row)
                i += 1
            make_table(doc, rows, total); continue
        if line.startswith("# "):
            words = line[2:]
            if words == "VisAge Project Documentation":
                p = doc.add_paragraph(words, style="Title")
                for run in p.runs:
                    run.font.color.rgb = BLACK; run.font.italic = False; run.font.name = "Calibri"
            elif words == "Contents":
                p = doc.add_paragraph(words); p.paragraph_format.keep_with_next = True
                run = p.runs[0]; run.bold = True; run.font.size = Pt(18)
            else:
                p = doc.add_heading(words, level=1)
                if words == "16 References":
                    p.paragraph_format.page_break_before = True
            i += 1; continue
        if line.startswith("## "):
            if first_h2:
                p = doc.add_paragraph(line[3:], style="Subtitle"); first_h2 = False
                for run in p.runs:
                    run.font.color.rgb = BLACK; run.font.italic = False; run.font.name = "Calibri"
            else:
                doc.add_heading(line[3:], level=2)
            i += 1; continue
        match = re.match(r"^(\d+)\. (.*)", line)
        if line.startswith("- ") or match:
            kind = bool(match)
            if current_list is None or current_list[0] != kind:
                current_list = (kind, list_id(doc, kind))
            p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(4)
            numpr = p._p.get_or_add_pPr().get_or_add_numPr(); numpr.get_or_add_ilvl().val = 0; numpr.get_or_add_numId().val = current_list[1]
            inline(p, match[2] if match else line[2:]); i += 1; continue
        p = doc.add_paragraph(style="Caption" if line.startswith("Figure ") else "Normal"); inline(p, line); i += 1
    for root in (doc.styles.element, doc.element):
        for border in root.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    doc.save(OUTPUT)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
