from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
 
llm = OpenAI(model='o3-mini-2025-01-31')

result =llm.invoke("Tell me a joke about programming.")

print(result)