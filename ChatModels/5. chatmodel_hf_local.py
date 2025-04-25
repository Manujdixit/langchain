from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

os.environment['HF_HOME']='D:/hugginggface_cache'

llm=HuggingFacePipeline.from_model_id(
    model_id="tinyllmama",
    task="text-generation",
    pipeline_kwargs = dict(
        temperature=0.5,
        max_new_token=100
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("what is capital of india?")

print(result)