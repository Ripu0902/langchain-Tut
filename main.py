import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env (if it exists)
load_dotenv()

# Ask for key if not found
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass(
        "Google API Key not found. Please enter your Google API Key: "
    )

# Initialize Gemini model (through LangChain)
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Run the model and store the output
response = model.invoke("Hello, world!")

# ✅ Print the model's text response
print(response)
