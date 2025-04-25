from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="TinyLLama",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is capital of india")

print(result.content)