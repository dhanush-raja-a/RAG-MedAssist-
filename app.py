from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from langchain_groq import ChatGroq

app =Flask(__name__)

load_dotenv()
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
PINCONE_API_KEY=os.environ.get("PINCONE_API_KEY")
os.environ["GROQ_API_KEY"]=GROQ_API_KEY
os.environ["PINECONE_API_KEY"]=PINCONE_API_KEY

embeddings=download_hugging_face_embeddings()


index_name = "medibotultra1"
data_to_model=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriver=data_to_model.as_retriever(search_type="similarity", search_kwargs={"k":3})

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_retries=2,)
Prompt= ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)
question_answer_chain=create_stuff_documents_chain(llm, Prompt)
rag_chain=create_retrieval_chain(retriver,question_answer_chain)

@app.route("/")  
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input": msg})
    print("response :", response["answer"])
    return str(response["answer"])

# Add this after app initialization
if os.environ.get('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
else:
    app.config['DEBUG'] = True

# Modify the last part
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port)
