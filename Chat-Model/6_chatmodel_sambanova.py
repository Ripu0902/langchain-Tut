from langchain.llms import SambaNova
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the SambaNova LLM
llm = SambaNova(
    endpoint_url=os.getenv("SAMBANOVA_ENDPOINT_URL"),
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    model_kwargs={
        "do_sample": True,
        "temperature": 0.7,
        "max_tokens": 100
    }
)

# Test the model with a simple prompt
prompt = "Explain what machine learning is in simple terms."
response = llm.invoke(prompt)

print(response)