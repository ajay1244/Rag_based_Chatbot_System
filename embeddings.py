from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Read PDF
reader = PdfReader("data/rag_oops.pdf")

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

print("Total Chunks:", len(chunks))

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Convert chunks to vectors
vectors = model.encode(chunks)

print("Embedding Shape:")
print(vectors.shape)