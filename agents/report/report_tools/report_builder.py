from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)

from agents.report.report_tools.report_design import create_styles, get_color, ALIGNMENTS

class ReportBuilder:
    def __init__(self):
        self.theme = {
            "font": "helvetica",
            "body_size": 11,
            "title_size": 26,
            "text_color": "#222222",
            "heading_color": "#1F3A5F",
            "accent_color": "#2E75B6",
            "line_spacing": 1.4,
            "page_size": "a4",
            "margin_cm": 2,
        }

        self.blocks = []
        self.header_text = ""
        self.footer_text = ""
        self.page_numbers = False

    def set_design(
        self,
        font=None,
        body_size=None,
        title_size=None,
        text_color=None,
        heading_color=None,
        accent_color=None,
        line_spacing=None,
        page_size=None,
        margin_cm=None
    ):
        if font:
            self.theme["font"] = font

        if body_size:
            self.theme["body_size"] = body_size

        if title_size:
            self.theme["title_size"] = title_size

        if text_color:
            self.theme["text_color"] = text_color

        if heading_color:
            self.theme["heading_color"] = heading_color

        if accent_color:
            self.theme["accent_color"] = accent_color

        if line_spacing:
            self.theme["line_spacing"] = line_spacing

        if page_size:
            self.theme["page_size"] = page_size

        if margin_cm:
            self.theme["margin_cm"] = margin_cm

    def set_header_footer(
        self,
        header_text=None,
        footer_text=None,
        page_numbers=False
    ):
        self.header_text = header_text or ""
        self.footer_text = footer_text or ""
        self.page_numbers = page_numbers

    def add_title(self, title, subtitle=""):
        styles = create_styles(self.theme)

        self.blocks.append(
            Paragraph(escape(title), styles["ReportTitle"])
        )

        if subtitle:
            self.blocks.append(
                Paragraph(escape(subtitle), styles["Subtitle"])
            )

        self.add_divider()

    def add_heading(self, text, level=1):

        styles = create_styles(self.theme)
        style_name = f"Heading{level}Custom"

        self.blocks.append(
            Paragraph(escape(text), styles[style_name])
        )

    def add_paragraph(
        self,
        text,
        bold=False,
        italic=False,
        align="left"
    ):
        styles = create_styles(self.theme)
        style = styles["Body"]

        if bold or italic:
            style = style.clone("CustomBody")

            if bold:
                style.fontName = self.theme["font"] + "-Bold"

            if italic:
                style.fontName = self.theme["font"] + "-Oblique"

        style.alignment = ALIGNMENTS[align]

        self.blocks.append(
            Paragraph(escape(text), style)
        )

    def add_bullets(self, items, numbered=False):
        styles = create_styles(self.theme)

        for index, item in enumerate(items, start=1):
            if numbered:
                prefix = f"{index}."
            else:
                prefix = "•"

            self.blocks.append(
                Paragraph(
                    f"{prefix} {escape(item)}",
                    styles["Body"]
                )
            )

    def add_table(self, headers, rows, header_color=None, zebra=True):
        styles = create_styles(self.theme)

        data = [headers] + rows
        table = Table(data, repeatRows=1)

        commands = [
            ("BACKGROUND", (0, 0), (-1, 0),
             get_color(header_color or self.theme["accent_color"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0),
             self.theme["font"] + "-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1),
             self.theme["body_size"]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]

        if zebra:
            for row_index in range(1, len(data)):
                if row_index % 2 == 0:
                    commands.append(
                        ("BACKGROUND", (0, row_index), (-1, row_index),
                         colors.HexColor("#F2F5F8"))
                    )

        table.setStyle(TableStyle(commands))

        self.blocks.append(table)
        self.blocks.append(Spacer(1, 10))

    def add_callout(self, text, background="#EAF2FB"):
        styles = create_styles(self.theme)

        table = Table(
            [[Paragraph(escape(text), styles["Body"])]],
            colWidths=[16 * cm]
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), get_color(background)),
            ("BOX", (0, 0), (-1, -1), 1, get_color(self.theme["accent_color"])),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))

        self.blocks.append(table)
        self.blocks.append(Spacer(1, 10))

    def add_divider(self):
        self.blocks.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=get_color(self.theme["accent_color"]),
                spaceAfter=10
            )
        )

    def add_space(self, height=12):
        self.blocks.append(Spacer(1, height))

    def add_page_break(self):
        self.blocks.append(PageBreak())

    # ---------------- PDF ----------------

    def build(self, path):
        page_sizes = {
            "a4": A4,
            "letter": LETTER
        }

        page_size = page_sizes[self.theme["page_size"]]
        margin = self.theme["margin_cm"] * cm

        doc = SimpleDocTemplate(
            str(path),
            pagesize=page_size,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
            title="Generated Report"
        )

        def draw_page(canvas, document):
            canvas.saveState()

            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.grey)

            if self.header_text:
                canvas.drawString(
                    margin,
                    page_size[1] - margin / 2,
                    self.header_text
                )

            if self.footer_text:
                canvas.drawString(
                    margin,
                    margin / 2,
                    self.footer_text
                )

            if self.page_numbers:
                canvas.drawRightString(
                    page_size[0] - margin,
                    margin / 2,
                    f"Page {document.page}"
                )

            canvas.restoreState()

        doc.build(
            self.blocks,
            onFirstPage=draw_page,
            onLaterPages=draw_page
        )

        return path
