from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# Ask question
response = model.generate_content(
    "Give me 5 interview questions on Python OOP."
)

print(response.text)