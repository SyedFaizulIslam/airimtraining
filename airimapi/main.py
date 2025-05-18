from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import uvicorn
from fastapi import FastAPI,File,HTTPException,UploadFile,Depends
import os
import boto3
import json
from Model import ChatPrompt
from dotenv import load_dotenv,find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import pandas as pd
import numpy as np
import sklearn
import joblib
from text_utils import clean_text
import tempfile
import PyPDF2
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from typing import Dict, List, Optional
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

@asynccontextmanager
async def modelinitialize(app: FastAPI):
    global model_llm, llm_embeddings, vectorizer_load, model_load, bedrock_runtime, bedrock_embeddings, chroma_client
    model_llm=ChatOpenAI(model_name="gpt-4o",temperature=0,api_key=OPENAI_API_KEY)
    llm_embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # Load the vectorizer and model
    VecorizerPath = "ConsumerComplainTfidfVectorizer.pkl"
    ModelPath = "ConsumerComplainModel.pkl"
    vectorizer_load = joblib.load(VecorizerPath)
    model_load = joblib.load(ModelPath)
    
    # Initialize AWS Bedrock client
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )
        print("AWS Bedrock client initialized successfully")
        
        # Initialize Bedrock embeddings for vector store
        bedrock_embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v2:0"  # Amazon's Titan embedding model
        )
        
        # Initialize Chroma DB client
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        chroma_client = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=bedrock_embeddings
        )
        print("Chroma DB client initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize AWS Bedrock or Chroma DB: {str(e)}")
        bedrock_runtime = None
        bedrock_embeddings = None
        chroma_client = None
    
    yield

app=FastAPI(lifespan=modelinitialize)

@app.post("/chat")
async def chat(chatprompt:ChatPrompt):
    result=model_llm.invoke(chatprompt.prompt)
    return result

@app.post("/qna")
async def qna(chatprompt:ChatPrompt):
    """
    QnA endpoint: Takes user question and returns answer using LangChain and OpenAI.
    """
    try:
        # Create a template that formats the question appropriately
        template = """
        You are a helpful assistant that provides clear and concise answers.
        
        Question: {question}
        
        Answer:
        """
        
        # Create a prompt template with LangChain
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        
        # Format the prompt with the user's question
        formatted_prompt = prompt_template.format(question=chatprompt.prompt)
        
        # Get response from the model
        response = model_llm.invoke(formatted_prompt)
        
        return {
            "question": chatprompt.prompt,
            "answer": response.content,
            "source": "OpenAI via LangChain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing QnA: {str(e)}")

@app.post("/bedrock-llama")
async def bedrock_llama(chatprompt: ChatPrompt):
    """
    Endpoint that invokes Llama 3 70B Instruct v1 on AWS Bedrock
    
    Parameters:
    - prompt: Text input to send to the model
    
    Returns:
    - Generated text from Llama 3
    """
    if not bedrock_runtime:
        raise HTTPException(
            status_code=503, 
            detail="AWS Bedrock client not initialized. Check your AWS credentials and configuration."
        )
    
    try:
        # Model ID for Llama 3 70B Instruct on Bedrock
        model_id = "arn:aws:bedrock:us-east-1:337608386354:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
        
        # Prepare request parameters
        request_body = {
            "prompt": chatprompt.prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9
        }
        
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json",
            performanceConfigLatency="standard"
        )
        
        # Parse the response
        response_body = json.loads(response.get("body").read())
        
        return {
            "prompt": chatprompt.prompt,
            "completion": response_body.get("generation", ""),
            "model": "Llama 3 70B Instruct v1"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error invoking AWS Bedrock: {str(e)}"
        )

@app.post("/predictconsumercompaint")
async def predictconsumercompaint(chatprompt:ChatPrompt):
    data = {'consumer_complaint_narrative': [chatprompt.prompt]}
    X=pd.DataFrame.from_dict(data)
    predict_features = vectorizer_load.transform(X["consumer_complaint_narrative"])
    y_pred = model_load.predict(predict_features)
    result=y_pred[0]
    return result

@app.post("/qlora_fine_tune")
async def qlora_fine_tune(chatprompt:ChatPrompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model_path = "Qlora_finetuned_model"
    # Load fine-tuned model
    model_finetune = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,    # or torch.float32 if no GPU
        device_map="auto"             # Automatically use GPU if available
    )
    tokenizer_finetune = AutoTokenizer.from_pretrained(model_path)
    # Again make sure pad_token is handled
    tokenizer_finetune.pad_token = tokenizer_finetune.eos_token
    # Example prompt
    prompt = chatprompt.prompt
    # Tokenize
    inputs = tokenizer_finetune(prompt, return_tensors="pt").to(model_finetune.device)
    # Generate output
    outputs = model_finetune.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    # Decode and print
    response = tokenizer_finetune.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), collection_name: Optional[str] = "default"):
    """
    Upload a PDF file, extract text, chunk it, and store embeddings in Chroma DB
    
    Parameters:
    - file: PDF file to upload and process
    - collection_name: Optional name for the Chroma collection (defaults to "default")
    
    Returns:
    - Information about the processed PDF and search results to verify embedding
    """
    if not bedrock_runtime or not bedrock_embeddings:
        raise HTTPException(
            status_code=503, 
            detail="AWS Bedrock not initialized. Check your AWS credentials and configuration."
        )
    
    try:
        # 1. Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # 2. Extract text from PDF
        pdf_text = ""
        file_size = os.path.getsize(temp_file_path)
        
        try:
            with open(temp_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_text += page.extract_text() + "\n\n"
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
        
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # 3. Create text chunks with RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        chunks = text_splitter.split_text(pdf_text)
        
        # 4. Add metadata to each chunk
        metadata = [
            {
                "filename": file.filename,
                "file_size": file_size,
                "chunk": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))
        ]
        
        # 5. Create Chroma collection and add documents
        collection = Chroma.from_texts(
            texts=chunks,
            embedding=bedrock_embeddings,
            metadatas=metadata,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # 6. Perform a quick search to verify embeddings
        if chunks:
            # Use the first chunk as a query to verify
            sample_query = chunks[0][:100]  # Use first 100 chars of first chunk
            search_results = collection.similarity_search(
                query=sample_query,
                k=2
            )
            
            # Extract results for response
            result_docs = []
            for doc in search_results:
                result_docs.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
        
        return {
            "status": "success",
            "filename": file.filename,
            "file_size": file_size,
            "total_pages": len(pdf_reader.pages) if 'pdf_reader' in locals() else 0,
            "chunks_created": len(chunks),
            "collection_name": collection_name,
            "sample_search_results": result_docs,
            "message": "PDF processed and embedded successfully"
        }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.post("/search_pdf")
async def search_pdf(chatprompt: ChatPrompt, collection_name: Optional[str] = "default", k: int = 3):
    """
    Search the Chroma DB using semantic search
    
    Parameters:
    - query: Text to search for
    - collection_name: Name of the collection to search
    - k: Number of results to return
    
    Returns:
    - Search results with document content and metadata
    """
    if not bedrock_embeddings:
        raise HTTPException(
            status_code=503,
            detail="Bedrock embeddings not available. Check AWS configuration."
        )
    
    try:
        # Load the collection
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=bedrock_embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # Perform similarity search
        results = collection.similarity_search(
            query=chatprompt.prompt,
            k=k
        )
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "query": chatprompt.prompt,
            "results": formatted_results,
            "collection_name": collection_name
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

@app.post("/query_document")
async def query_document(chatprompt: ChatPrompt, collection_name: Optional[str] = "default", k: int = 5):
    """
    Query documents from Chroma DB and get intelligent responses using Llama 4 Scout
    
    Parameters:
    - prompt: User's question about the documents
    - collection_name: Name of the Chroma collection to search
    - k: Number of most relevant chunks to retrieve
    
    Returns:
    - Intelligent response based on the retrieved document chunks
    """
    if not bedrock_runtime or not bedrock_embeddings:
        raise HTTPException(
            status_code=503,
            detail="AWS Bedrock or embeddings not initialized. Check AWS configuration."
        )
    
    try:
        # 1. Load the Chroma collection
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=bedrock_embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # 2. Retrieve relevant document chunks
        relevant_docs = collection.similarity_search(
            query=chatprompt.prompt,
            k=k
        )
        
        # 3. Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 4. Prepare metadata for response
        sources = []
        for doc in relevant_docs:
            if doc.metadata and "filename" in doc.metadata:
                source = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk": doc.metadata.get("chunk", "Unknown")
                }
                sources.append(source)
        
        # 5. Create prompt for Llama 4 Scout - format appropriately for the model
        formatted_prompt = f"""<system>
You are a helpful AI assistant that answers questions based on the provided document context. 
Your answers should be comprehensive, accurate, and based solely on the information provided in the context. 
If the context doesn't contain enough information to answer the question fully, acknowledge this limitation.
Cite specific parts of the context to support your answer when appropriate.
</system>

<user>
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question: {chatprompt.prompt}
</user>

<assistant>
"""
        
        # 6. Invoke Llama 4 Scout model
        model_id = "arn:aws:bedrock:us-east-1:337608386354:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0"
        
        # Format request body according to Llama 4 Scout specifications - using prompt parameter instead of messages
        request_body = {
            "prompt": formatted_prompt,
            "max_gen_len": 1024,
            "temperature": 0.2,  # Lower temperature for more factual responses
            "top_p": 0.9
        }
        
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json",
            performanceConfigLatency="standard"
        )
        
        # 7. Parse the response
        response_body = json.loads(response.get("body").read())
        
        # Extract the generated text from the response based on Llama 4 Scout's response format
        generated_text = ""
        if "generation" in response_body:
            generated_text = response_body["generation"]
        else:
            # Fall back to string representation of response if format is unexpected
            generated_text = str(response_body)
        
        # 8. Return the response with sources
        return {
            "question": chatprompt.prompt,
            "answer": generated_text,
            "sources": sources,
            "model": "Llama 4 Scout 17B",
            "total_chunks_retrieved": len(relevant_docs)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying documents: {str(e)}"
        )

if __name__ == '__main__':
    #load_models()
    
    uvicorn.run(app,port=8000)