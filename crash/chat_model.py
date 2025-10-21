from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
import os

load_dotenv()


llm_openai = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


PROJECT_ID = os.environ.get("PROJECT_ID")
SESSION_ID = "test_session_001"
COLLECTION_NAME = "chat_history"

print("Initializing fire store client....")
client = firestore.Client(project=PROJECT_ID)

chat_history = FirestoreChatMessageHistory(
    client=client,
    collection=COLLECTION_NAME,
    session_id=SESSION_ID
)

print("Chat History Initialized")
print("Current Chat History:,", chat_history.messages)


system_message = SystemMessage(content="You are a helpful AI assistant.")

chat_history.append(system_message)

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        break
    chat_history.add_user_message(query)
    
    result = llm_gemini.invoke(chat_history.messages)
    response = result.content
    print(f"AI: {response}")
    chat_history.add_ai_message(result.content)


print("Chat ended.")

print("Final chat history:")
for message in chat_history:
    print(f"{message.type}: {message.content}")