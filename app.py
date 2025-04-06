from urllib.parse import quote as url_quote  # Correct import for URL encoding
from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from langchain_groq import ChatGroq
from src.helper import get_stock_info, get_mutual_fund_info
import re

app = Flask(__name__)

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINCONE_API_KEY = os.environ.get("PINCONE_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINCONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medibotultra1"
data_to_model = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriver = data_to_model.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_retries=2,
)
Prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, Prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

def extract_stock_symbol(text):
    """Extract stock symbol from user message"""
    # Common patterns for stock queries
    patterns = [
        r"stock price of (\w+)",
        r"(\w+) stock",
        r"price of (\w+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).upper()
    return None

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    
    # Check if it's a stock price query
    stock_symbol = extract_stock_symbol(msg)
    if stock_symbol:
        stock_data = get_stock_info(stock_symbol)
        if stock_data:
            response = f"Here's the current information for {stock_data['name']}:\n"
            response += f"Current Price: ₹{stock_data['current_price']}\n"
            response += f"Previous Close: ₹{stock_data['prev_close']}\n"
            response += f"Day High: ₹{stock_data['day_high']}\n"
            response += f"Day Low: ₹{stock_data['day_low']}"
            return response
    
    # If not a stock query, use the RAG chain
    response = rag_chain.invoke({"input": msg})
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
