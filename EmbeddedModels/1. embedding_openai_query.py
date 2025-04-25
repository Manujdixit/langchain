from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents=[
    "delhi is capital of india",
    "kolkakta is capital of west bengal",
    "paris is capital of france"
]

result=embedding.embed_documents(documents)

print(str(result))