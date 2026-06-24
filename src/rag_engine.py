from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def generate_answer(query, chunks, index, embedding_model):

    query_vector = embedding_model.encode([query])

    import numpy as np

    query_vector = np.array(query_vector).astype("float32")

    D, I = index.search(query_vector, k=5)

    retrieved_text = ""

    for idx in I[0]:
        retrieved_text += chunks[idx]
        retrieved_text += "\n\n"

    prompt = f"""
Use only the context below.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"