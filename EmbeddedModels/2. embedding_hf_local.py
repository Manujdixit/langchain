from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='model_name')

text="dehli is capital oof india"

vector=embedding.embed_query(text)

print(str(vector))