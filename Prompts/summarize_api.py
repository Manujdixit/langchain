from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

load_dotenv()

app=FastAPI(
    title="Prompt UI",
    description="A simple UI for prompting with LangChain",
    version="0.1.0",
)

model=ChatOpenAI("gpt-3.5-turbo")

template=load_prompt("template.json")

class SummarizeRequest(BaseModel):
    paper_input: str
    style_input: str
    length_input: str

    class Config:
        schema_extra = {
            "example": {
                "paper_input": "The paper titled 'The Effects of Climate Change on Agriculture' by John Smith",
                "style_input": "Technical",
                "length_input": "Concise"
            }
        }

class SummarizeResponse(BaseModel):
    summary: str
    metadata: dict

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    try:

        chain=template|model

        result=chain.invoke({
            'paper_input':request.paper_input,
            'style_input':request.style_input,
            'length_input':request.length_input
        })

        return SummarizeResponse(summary=result.content,metadata={
            "paper_input": request.paper_input,
            "style_input": request.style_input,
            "length_input": request.length_input
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 