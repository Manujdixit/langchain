from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents=[
    "virat is a cricketer",
    "ronaldo cooks",
    "messi is a goat"
]

query="tell me about virat kohli"

doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

#[0] because we dont want 2d list, we want simple array.
scores=cosine_similarity([query_embedding],doc_embeddings)[0]
#response-
# [(0,0.123456),(1,0.3456789),(2,0.32547),(3,0.98745)]

#sort according to second element
index,score=(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])

print(query)
print(documents[index])
print("similarity score is:",score)