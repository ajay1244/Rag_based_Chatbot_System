from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------
# Load PDF
# -------------------------
reader = PdfReader("Data/rag_oops.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# -------------------------
# Chunking
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

# -------------------------
# Embeddings
# -------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

vectors = model.encode(chunks)

vectors = np.array(vectors).astype("float32")

# -------------------------
# FAISS
# -------------------------
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

# -------------------------
# User Question
# -------------------------
query = "What is inheritance?"

query_vector = model.encode([query])

query_vector = np.array(query_vector).astype("float32")

# Search Top 3
D, I = index.search(query_vector, k=3)

print("Top Matching Chunks:\n")

for idx in I[0]:
    print("=" * 50)
    print(chunks[idx])
    print()