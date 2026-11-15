def format_sources(evidence):

    sources = []
    context = []

    for index, item in enumerate(evidence, start=1):

        # Handle LangChain Document objects
        if hasattr(item, "page_content"):

            content = item.page_content
            metadata = item.metadata

        elif isinstance(item, dict):

            content = item.get(
                "page_content",
                item.get("text", item.get("content", ""))
            )

            metadata = item.get("metadata", item)

        else:
            content = str(item)
            metadata = {}

        document_name = metadata.get(
            "document_name",
            metadata.get("doc_name",
                metadata.get("source", metadata.get("filename", "Unknown document")))
        )

        page_number = metadata.get(
            "page_number",
            metadata.get("page")
        )

        source_id = f"S{index}"

        sources.append({
            "id": source_id,
            "document_name": document_name,
            "page_number": page_number
        })

        context.append(
            f"[{source_id}]\n"
            f"Document: {document_name}\n"
            f"Page: {page_number}\n"
            f"Content: {content}"
        )

    return {
        "sources": sources,
        "context": "\n\n".join(context)
    }
