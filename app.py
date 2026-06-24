from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.embedder import create_embeddings, model
from src.vector_store import build_index
from src.rag_engine import generate_answer

# Load PDF
text = load_pdf("Data/rag_oops.pdf")

# Create Chunks
chunks = create_chunks(text)

# Embeddings
vectors = create_embeddings(chunks)

# FAISS
index = build_index(vectors)

print(f"Total Chunks: {len(chunks)}")

while True:

    query = input("\nAsk Question (exit to quit): ")

    if query.lower() == "exit":
        break

    answer = generate_answer(
        query,
        chunks,
        index,
        model
    )

    print("\nAnswer:\n")
    print(answer)