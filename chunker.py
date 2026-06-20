from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read PDF
reader = PdfReader("Data/rag_oops.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

print(f"Total Characters: {len(text)}")

# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"\nTotal Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0])