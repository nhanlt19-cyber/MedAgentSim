import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def apply_inline_formatting(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(apply_inline_formatting(text), style)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def build_story(markdown_text: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletBody",
        parent=body,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=0,
    )
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, spaceAfter=8)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=12, leading=16, spaceAfter=6)

    story = []
    lines = markdown_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 0.15 * cm))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            if rows:
                table_data = [[paragraph(cell, body) for cell in row] for row in rows]
                tbl = Table(table_data, repeatRows=1)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAF6")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 0.2 * cm))
            continue

        if stripped.startswith("- "):
            bullet_items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item_text = lines[i].strip()[2:].strip()
                bullet_items.append(ListItem(paragraph(item_text, bullet_style)))
                i += 1
            story.append(ListFlowable(bullet_items, bulletType="bullet", leftIndent=18))
            story.append(Spacer(1, 0.15 * cm))
            continue

        if stripped.startswith("# "):
            story.append(paragraph(stripped[2:].strip(), h1))
            i += 1
            continue
        if stripped.startswith("## "):
            story.append(paragraph(stripped[3:].strip(), h2))
            i += 1
            continue
        if stripped.startswith("### "):
            story.append(paragraph(stripped[4:].strip(), h3))
            i += 1
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "-", "|")):
                break
            para_lines.append(nxt)
            i += 1
        story.append(paragraph(" ".join(para_lines), body))

    return story


def convert_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    markdown_text = input_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )
    story = build_story(markdown_text)
    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a simple Markdown report to PDF.")
    parser.add_argument("--input-md", required=True, help="Path to the Markdown file.")
    parser.add_argument("--output-pdf", required=True, help="Path to the output PDF file.")
    args = parser.parse_args()

    convert_markdown_to_pdf(Path(args.input_md).resolve(), Path(args.output_pdf).resolve())
    print(f"Wrote {Path(args.output_pdf).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
