def select_context(documents, max_chars=12000):

    selected_documents = []
    seen_content = set()
    total_chars = 0

    for document in documents:

        content = document.page_content.strip()

        if not content or content in seen_content:
            continue

        if total_chars + len(content) > max_chars:
            continue

        selected_documents.append(document)
        seen_content.add(content)
        total_chars += len(content)

    return selected_documents


def format_context(documents):

    context = []

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata

        source = metadata.get("document_name", "Unknown document")
        page = metadata.get("page_number", "Unknown")

        context.append(
            f"""
            [Source {index}]
            Document: {source}
            Page: {page}

            Content:
            {document.page_content}
            """
        )

    return "\n".join(context)
