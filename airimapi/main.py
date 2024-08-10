from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
import uvicorn
from fastapi import FastAPI,File,HTTPException,UploadFile,Depends
import os
from Model import ChatPrompt
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@asynccontextmanager
async def modelinitialize(app: FastAPI):
    global model_llm
    model_llm=ChatOpenAI(model_name="gpt-4o",temperature=0,api_key=OPENAI_API_KEY)
    yield

app=FastAPI(lifespan=modelinitialize)

@app.post("/chat")
async def chat(chatprompt:ChatPrompt):
    result=model_llm.invoke(chatprompt.prompt)
    return result

if __name__ == '__main__':
    #load_models()
    uvicorn.run(app,port=8000)