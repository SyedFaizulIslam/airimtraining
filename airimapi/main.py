from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import uvicorn
from fastapi import FastAPI,File,HTTPException,UploadFile,Depends
import os
from Model import ChatPrompt
from dotenv import load_dotenv,find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma,MongoDBAtlasVectorSearch
import pandas as pd
import numpy as np
import sklearn
import joblib
from text_utils import clean_text
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@asynccontextmanager
async def modelinitialize(app: FastAPI):
    global model_llm,llm_embeddings,vectorizer_load,model_load
    model_llm=ChatOpenAI(model_name="gpt-4o",temperature=0,api_key=OPENAI_API_KEY)
    llm_embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # Load the vectorizer and model
    VecorizerPath = "ConsumerComplainTfidfVectorizer.pkl"
    ModelPath = "ConsumerComplainModel.pkl"
    vectorizer_load = joblib.load(VecorizerPath)
    model_load = joblib.load(ModelPath)
    yield

app=FastAPI(lifespan=modelinitialize)

@app.post("/chat")
async def chat(chatprompt:ChatPrompt):
    result=model_llm.invoke(chatprompt.prompt)
    return result

@app.post("/predictconsumercompaint")
async def predictconsumercompaint(chatprompt:ChatPrompt):
    data = {'consumer_complaint_narrative': [chatprompt.prompt]}
    X=pd.DataFrame.from_dict(data)
    predict_features = vectorizer_load.transform(X["consumer_complaint_narrative"])
    y_pred = model_load.predict(predict_features)
    result=y_pred[0]
    return result

if __name__ == '__main__':
    #load_models()
    
    uvicorn.run(app,port=8000)