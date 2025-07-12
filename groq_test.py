import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
model = genai.GenerativeModel('gemini-2.0-flash')

# Generate content
response = model.generate_content("Hello, can you help me plan a trip?")

print(response.text)
