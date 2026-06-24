import streamlit as st
import tempfile
from pypdf import PdfReader

from src.chunker import create_chunks
from src.embedder import create_embeddings, model
from src.vector_store import build_index
from src.rag_engine import generate_answer
from src.embedder import create_embeddings, model

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI PDF Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

        chunks = create_chunks(text)

        vectors = create_embeddings(chunks)

        index = build_index(vectors)

    st.success(
        f"PDF Processed Successfully! Chunks: {len(chunks)}"
    )

    query = st.text_input(
        "Ask a question from the PDF"
    )

if st.button("Get Answer"):

    if query:

        try:

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    query,
                    chunks,
                    index,
                    model
                )

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )