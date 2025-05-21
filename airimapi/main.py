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

@app.post("/QnA_From_Llama")
async def qna_from_llama(chatprompt: ChatPrompt):
    """
    QnA endpoint that uses Llama 3 70B on AWS Bedrock to answer questions
    
    Parameters:
    - prompt: The question to ask the model
    
    Returns:
    - A structured response with the question and answer from Llama 3
    """
    if not bedrock_runtime:
        raise HTTPException(
            status_code=503, 
            detail="AWS Bedrock client not initialized. Check your AWS credentials and configuration."
        )
    
    try:
        # Format the prompt for better QnA structuring
        formatted_prompt = f"""<system>
You are a helpful AI assistant that provides informative, accurate, and helpful answers to user questions.
Respond in a clear and concise manner, ensuring your answer addresses the specific question asked.
</system>

<user>
{chatprompt.prompt}
</user>

<assistant>
"""
        
        # Set up model parameters - similar to the CLI command
        model_id = "arn:aws:bedrock:us-east-1:337608386354:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
        
        request_body = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9
        }
        
        # Invoke the model using the API
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json",
            performanceConfigLatency="standard"
        )
        
        # Process response
        response_body = json.loads(response.get("body").read())
        
        # Extract the generated text
        if "generation" in response_body:
            answer = response_body["generation"]
        else:
            # Handle unexpected response format
            answer = "Unable to parse model response: " + str(response_body)
        
        # Return structured response
        return {
            "question": chatprompt.prompt,
            "answer": answer,
            "model": "Llama 3 70B Instruct v1",
            "metadata": {
                "temperature": 0.5,
                "top_p": 0.9,
                "max_gen_len": 512
            }
        }
        
    except Exception as e:
        # Provide detailed error information
        error_message = str(e)
        
        # Check for specific AWS errors
        if "ValidationException" in error_message:
            detail = "Request format error: " + error_message
        elif "AccessDeniedException" in error_message:
            detail = "AWS Bedrock access denied. Check your credentials and permissions."
        elif "ResourceNotFoundException" in error_message:
            detail = "Model not found. Check the model ARN."
        else:
            detail = f"Error querying Llama model: {error_message}"
            
        raise HTTPException(
            status_code=500,
            detail=detail
        )

@app.post("/Demo_Pdf_embeddings")
async def demo_pdf_embeddings(file: UploadFile = File(...)):
    """
    Demo endpoint to upload a PDF file, chunk it into 500-character pieces,
    embed using Amazon Titan model, and store in Chroma DB
    
    Parameters:
    - file: PDF file to upload and process
    
    Returns:
    - Success message with filename and embedding details
    """
    if not bedrock_runtime or not bedrock_embeddings:
        raise HTTPException(
            status_code=503, 
            detail="AWS Bedrock not initialized. Check your AWS credentials and configuration."
        )
    
    try:
        # Create a unique collection name based on timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = f"demo_pdf_{timestamp}"
        
        # 1. Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # 2. Extract text from PDF
        pdf_text = ""
        file_size = os.path.getsize(temp_file_path)
        pdf_page_count = 0
        
        try:
            with open(temp_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pdf_page_count = len(pdf_reader.pages)
                for page_num in range(pdf_page_count):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + "\n\n"
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
        
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. The file might be empty or corrupted.")
        
        # 3. Create text chunks with specified size of 500 characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(pdf_text)
        
        # 4. Add metadata to each chunk
        metadata = [
            {
                "filename": file.filename,
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 2),
                "page_count": pdf_page_count,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "embedding_model": "amazon.titan-embed-text-v2:0",
                "created_at": timestamp
            } for i in range(len(chunks))
        ]
        
        # 5. Create Chroma collection and add documents with their embeddings
        collection = Chroma.from_texts(
            texts=chunks,
            embedding=bedrock_embeddings,  # This uses the amazon.titan-embed-text-v2:0 model
            metadatas=metadata,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # No need to call persist() as the from_texts() method with persist_directory already saves the data
        
        # 6. Return a detailed success message
        return {
            "status": "success",
            "message": f"PDF '{file.filename}' has been successfully processed and embedded",
            "details": {
                "filename": file.filename,
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 2),
                "page_count": pdf_page_count,
                "total_text_length": len(pdf_text),
                "chunks_created": len(chunks),
                "collection_name": collection_name,
                "embedding_model": "amazon.titan-embed-text-v2:0",
                "storage_location": CHROMA_PERSIST_DIRECTORY
            }
        }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF for embedding: {str(e)}"
        )

@app.post("/demo_query_pdf")
async def demo_query_pdf(
    chatprompt: ChatPrompt, 
    collection_name: Optional[str] = None, 
    k: int = 5
):
    """
    Query uploaded documents in Chroma DB and extract answers using Llama 3 70B model
    
    Parameters:
    - prompt: Question to ask about the documents
    - collection_name: Name of collection to query (if None, uses the most recent collection)
    - k: Number of relevant chunks to retrieve
    
    Returns:
    - Answer extracted from the embedded documents using Llama 3
    """
    if not bedrock_runtime or not bedrock_embeddings:
        raise HTTPException(
            status_code=503, 
            detail="AWS Bedrock not initialized. Check your AWS credentials and configuration."
        )
    
    try:
        # If no collection specified, try to find the most recent demo collection
        if not collection_name:
            # Get list of existing Chroma collections
            try:
                # Approach 1: Try to get existing collections directly from Chroma
                from chromadb import PersistentClient
                chroma_client = PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
                all_collections = chroma_client.list_collections()
                collections = [c.name for c in all_collections if c.name.startswith("demo_pdf_")]
                
                if not collections:
                    # Approach 2: If no demo collections found above, try directory scanning
                    collections_dir = os.path.join(CHROMA_PERSIST_DIRECTORY, "collections")
                    if os.path.exists(collections_dir):
                        collections = [d for d in os.listdir(collections_dir) if d.startswith("demo_pdf_")]
            except Exception as e:
                print(f"Error accessing collections: {e}")
                # Try direct path approach as fallback
                collections_dir = os.path.join(CHROMA_PERSIST_DIRECTORY, "collections")
                if os.path.exists(collections_dir):
                    collections = [d for d in os.listdir(collections_dir) if d.startswith("demo_pdf_")]
                else:
                    collections = []
            
            if not collections:
                # Try other common locations where collections might be stored
                possible_paths = [
                    os.path.join(CHROMA_PERSIST_DIRECTORY),
                    os.path.join(os.getcwd(), "chroma_db"),
                    os.path.join(os.getcwd(), "chroma_db", "collections")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        print(f"Checking path: {path}")
                        try:
                            # Try to list any collection-like directories
                            dirs = os.listdir(path)
                            demo_collections = [d for d in dirs if d.startswith("demo_pdf_")]
                            if demo_collections:
                                collections = demo_collections
                                print(f"Found collections in {path}: {collections}")
                                break
                        except:
                            continue
            
            if not collections:
                # Get all available collections as a fallback
                try:
                    # Use PersistentClient to list all collections
                    from chromadb import PersistentClient
                    client = PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
                    available_collections = client.list_collections()
                    collection_names = [c.name for c in available_collections]
                    
                    if collection_names:
                        # If we found any collections, use the first one
                        collection_name = collection_names[0]
                        print(f"Using fallback collection: {collection_name}")
                    else:
                        raise HTTPException(
                            status_code=404,
                            detail="No collections found in Chroma DB. Please upload a document first using Demo_Pdf_embeddings endpoint."
                        )
                except Exception as e:
                    print(f"Error listing collections: {e}")
                    raise HTTPException(
                        status_code=404,
                        detail="No document collections found. Please upload a document first using Demo_Pdf_embeddings endpoint."
                    )
            else:
                # Sort by timestamp in the name to get the most recent one
                collections.sort(reverse=True)
                collection_name = collections[0]
                print(f"Using collection: {collection_name}")
            
        # Load the specified Chroma collection
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=bedrock_embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # Search for relevant document chunks
        relevant_docs = collection.similarity_search(
            query=chatprompt.prompt,
            k=k
        )
        
        if not relevant_docs:
            return {
                "question": chatprompt.prompt,
                "answer": "No relevant documents found in the collection.",
                "collection_name": collection_name
            }
        
        # Compile document context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Collect source information
        sources = []
        for doc in relevant_docs:
            if doc.metadata and "filename" in doc.metadata:
                source = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", doc.metadata.get("chunk", "Unknown")),
                    "page_count": doc.metadata.get("page_count", "Unknown")
                }
                sources.append(source)
        
        # Create prompt for Llama 3
        formatted_prompt = f"""<system>
You are a helpful AI assistant that answers questions based only on the provided document context.
Your task is to extract answers directly from the documents without adding external information.
If the answer cannot be found in the document context, simply state that the information is not 
available in the provided documents.
</system>

<user>
DOCUMENT CONTEXT:
-----------------
{context}
-----------------

QUESTION: {chatprompt.prompt}

Answer the question based strictly on the information in the document context. Do not include any information from outside the context.
</user>

<assistant>
"""
        
        # Set up Llama 3 model request
        model_id = "arn:aws:bedrock:us-east-1:337608386354:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
        
        request_body = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.3,  # Lower temperature for more factual responses
            "top_p": 0.9
        }
        
        # Call Bedrock with Llama 3 70B model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json",
            performanceConfigLatency="standard"
        )
        
        # Parse response
        response_body = json.loads(response.get("body").read())
        
        # Extract answer from response
        if "generation" in response_body:
            answer = response_body["generation"]
        else:
            answer = "Unable to parse model response: " + str(response_body)
        
        # Return structured response
        return {
            "question": chatprompt.prompt,
            "answer": answer,
            "sources": sources,
            "collection_name": collection_name,
            "model": "Llama 3 70B Instruct v1",
            "chunks_retrieved": len(relevant_docs)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying PDF documents: {str(e)}"
        )

if __name__ == '__main__':
    #load_models()
    
    uvicorn.run(app,port=8000)