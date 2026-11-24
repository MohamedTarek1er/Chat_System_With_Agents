from typing import Literal
from langchain_core.tools import tool

def make_report_tools(builder):

    @tool
    def set_design(
        font: Literal["helvetica", "times", "courier"] = "helvetica",
        body_size: int = 11,
        title_size: int = 26,
        text_color: str = "#222222",
        heading_color: str = "#1F3A5F",
        accent_color: str = "#2E75B6",
        line_spacing: float = 1.4,
        page_size: Literal["a4", "letter"] = "a4",
        margin_cm: float = 2
    ):
        """Set report fonts, colors, sizes, spacing and page layout."""

        builder.set_design(
            font=font,
            body_size=body_size,
            title_size=title_size,
            text_color=text_color,
            heading_color=heading_color,
            accent_color=accent_color,
            line_spacing=line_spacing,
            page_size=page_size,
            margin_cm=margin_cm
        )

        return "Design updated successfully."

    @tool
    def set_header_footer(
        header_text: str = "",
        footer_text: str = "",
        page_numbers: bool = True
    ):
        """Set the header, footer, and page numbers."""

        builder.set_header_footer(
            header_text=header_text,
            footer_text=footer_text,
            page_numbers=page_numbers
        )

        return "Header and footer updated."

    @tool
    def add_title(title: str, subtitle: str = ""):
        """Add the report title and an optional subtitle."""

        builder.add_title(title, subtitle)

        return "Title added."

    @tool
    def add_heading(
        text: str,
        level: Literal[1, 2, 3] = 1
    ):
        """Add a heading. Level 1 is main, 2 is subsection, 3 is smaller."""

        builder.add_heading(text, level)

        return "Heading added."

    @tool
    def add_paragraph(
        text: str,
        bold: bool = False,
        italic: bool = False,
        align: Literal["left", "center", "right", "justify"] = "left"
    ):
        """Add a normal paragraph."""

        builder.add_paragraph(
            text,
            bold=bold,
            italic=italic,
            align=align
        )

        return "Paragraph added."

    @tool
    def add_bullets(
        items: list[str],
        numbered: bool = False
    ):
        """Add bullet points or a numbered list."""

        builder.add_bullets(items, numbered)

        return "List added."

    @tool
    def add_table(
        headers: list[str],
        rows: list[list[str]],
        header_color: str = "#2E75B6",
        zebra: bool = True
    ):
        """Add a table with headers and rows."""

        builder.add_table(
            headers,
            rows,
            header_color,
            zebra
        )

        return "Table added."

    @tool
    def add_callout(
        text: str,
        background: str = "#EAF2FB"
    ):
        """Add a highlighted box for an important finding or summary."""

        builder.add_callout(text, background)

        return "Callout added."

    @tool
    def add_divider():
        """Add a horizontal separator."""

        builder.add_divider()

        return "Divider added."

    @tool
    def add_space(height: int = 12):
        """Add vertical space between elements."""

        builder.add_space(height)

        return "Space added."

    @tool
    def add_page_break():
        """Start a new page."""

        builder.add_page_break()

        return "Page break added."

    return [
        set_design,
        set_header_footer,
        add_title,
        add_heading,
        add_paragraph,
        add_bullets,
        add_table,
        add_callout,
        add_divider,
        add_space,
        add_page_break
    ]
