from document_processing.document_loader import load_document
from document_processing.ingestion import ingest_documents
from document_processing.vector_store import search_documents

# Load multiple documents
documents = []

documents.extend(
    load_document(r"C:\Users\moham\Downloads\College\Graduation_Project\Papers\HUGS\Zhou_HUGS_Holistic_Urban_3D_Scene_Understanding_via_Gaussian_Splatting_CVPR_2024_paper.pdf")
)

documents.extend(
    load_document(r"C:\Users\moham\Downloads\College\Graduation_Project\Papers\Guide Dog\2501.07957v2.pdf")
)

documents.extend(
    load_document(r"C:\Users\moham\Downloads\College\Graduation_Project\Papers\DL3DV (Scene Narration)\Scene Narration.docx")
)

# Ingest documents into ChromaDB
total_chunks = ingest_documents(documents)

print(f"Successfully stored {total_chunks} chunks.")


# Test semantic search
results = search_documents(
    query="What are the models and their accuracy?",
    k=4,
)

for result in results:

    print("\nContent:")
    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)

    print("-" * 50)
