
from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

data=load_pdf_file(data="/Users/dhanushrajaa/Desktop/medibot/End-to-End-medibot/data")
chunk=text_split(data)
embeddings=download_hugging_face_embeddings()
PINCONE_API_KEY=os.environ.get("PINCONE_API_KEY")
pc=Pinecone(api_key=PINCONE_API_KEY)
index_name = "medibotultra1"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
os.environ["PINECONE_API_KEY"]=PINCONE_API_KEY

docsearch=PineconeVectorStore.from_documents(
    documents=chunk,
    index_name=index_name,
    embedding=embeddings,
)




