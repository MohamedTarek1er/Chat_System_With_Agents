from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import cm

FONTS = {
    "helvetica": "Helvetica",
    "times": "Times-Roman",
    "courier": "Courier",
}

ALIGNMENTS = {
    "left": TA_LEFT,
    "center": TA_CENTER,
    "right": TA_RIGHT,
    "justify": TA_JUSTIFY,
}

def get_color(value):
    return colors.HexColor(value)

def create_styles(theme):

    font = FONTS[theme["font"]]
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontName=font + "-Bold",
        fontSize=theme["title_size"],
        textColor=get_color(theme["heading_color"]),
        alignment=TA_CENTER,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name="Subtitle",
        fontName=font,
        fontSize=12,
        textColor=get_color(theme["accent_color"]),
        alignment=TA_CENTER,
        spaceAfter=15
    ))

    styles.add(ParagraphStyle(
        name="Body",
        fontName=font,
        fontSize=theme["body_size"],
        leading=theme["body_size"] * theme["line_spacing"],
        textColor=get_color(theme["text_color"]),
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="Heading1Custom",
        fontName=font + "-Bold",
        fontSize=18,
        textColor=get_color(theme["heading_color"]),
        spaceBefore=12,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="Heading2Custom",
        fontName=font + "-Bold",
        fontSize=14,
        textColor=get_color(theme["heading_color"]),
        spaceBefore=10,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="Heading3Custom",
        fontName=font + "-Bold",
        fontSize=12,
        textColor=get_color(theme["heading_color"]),
        spaceBefore=8,
        spaceAfter=5
    ))

    return styles
