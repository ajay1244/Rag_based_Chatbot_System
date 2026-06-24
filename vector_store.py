from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Read PDF
reader = PdfReader("Data/rag_oops.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = model.encode(chunks)

# Convert to numpy float32
vectors = np.array(vectors).astype("float32")

# Create FAISS index
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

print("Total vectors stored:", index.ntotal)