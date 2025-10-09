import streamlit as st
from sambanova import SambaNova
from dotenv import load_dotenv
import os
load_dotenv()



client = SambaNova(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)


st.header("DeepSeek-V3.1-Terminus")

USER_INPUT = st.text_input("Enter your message:", "How many stars are there in the Milky Way galaxy?") 

response = client.chat.completions.create(
    model="DeepSeek-V3.1-Terminus",
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":""+USER_INPUT}],
    temperature=0.1,
    top_p=0.1
)

if st.button("Send"):
    result = response.choices[0].message.content
    st.text("DeepSeek: " + result)

