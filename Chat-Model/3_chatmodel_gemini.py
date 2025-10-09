from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, max_output_tokens=100)
result = model.invoke("Write 5 line about life?")

print(result.content)