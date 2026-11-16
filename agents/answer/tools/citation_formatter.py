import re

def format_citations(response, sources):

    source_map = {
        source["id"]: source
        for source in sources
    }

    def replace_citation(match):

        source_id = "S" + match.group(1)
        source = source_map.get(source_id)

        if not source:
            return "[Unverified source]"

        document_name = source["document_name"]
        page_number = source["page_number"]

        if page_number is not None:
            return f"[{document_name}, p. {page_number}]"

        return f"[{document_name}]"

    response = re.sub(
        r"\[S(\d+)\]",
        replace_citation,
        response
    )

    source_list = []
    seen = set()

    for source in sources:

        key = (
            source["document_name"],
            source["page_number"]
        )

        if key in seen:
            continue

        seen.add(key)

        if source["page_number"] is not None:
            source_list.append(
                f"- {source['document_name']}, "
                f"Page {source['page_number']}"
            )
        else:
            source_list.append(
                f"- {source['document_name']}"
            )

    if source_list:
        response += "\n\n## Sources\n"
        response += "\n".join(source_list)

    return response
