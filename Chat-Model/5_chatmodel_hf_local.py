from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline



llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 150, "temperature": 0.01, "top_p": 0.95},
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("How many stars are there in the Milky Way galaxy?")

print(result.content)

