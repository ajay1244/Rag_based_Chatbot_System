from ollama import chat
import numpy as np


def generate_answer(query, chunks, index, embedding_model):

    # Convert question to embedding
    query_vector = embedding_model.encode([query])

    query_vector = np.array(query_vector).astype("float32")

    # Search top 5 similar chunks
    D, I = index.search(query_vector, k=6)

    # Hallucination prevention
    if D[0][0] > 1.5:
        return "The answer is not available in the provided document."

    retrieved_text = ""

    for idx in I[0]:
        retrieved_text += chunks[idx]
        retrieved_text += "\n\n"

    prompt = f"""
You are a Retrieval Augmented Generation (RAG) assistant.

STRICT RULES:

1. Use ONLY the provided context.
2. Do NOT use your own knowledge.
3. Do NOT make assumptions.
4. If the answer is not explicitly present in the context, respond EXACTLY:

The answer is not available in the provided document.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""

    try:

        response = chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"Error: {str(e)}"