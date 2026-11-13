def filter_by_metadata(documents, filters=None):

    if not filters:
        return documents

    filtered_documents = []

    for document in documents:

        metadata = document.metadata
        matches = True

        for key, value in filters.items():
            metadata_value = metadata.get(key)

            if isinstance(value, list):
                if metadata_value not in value:
                    matches = False
                    break

            elif metadata_value != value:
                matches = False
                break

        if matches:
            filtered_documents.append(document)

    return filtered_documents
