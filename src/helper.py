from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_pdf_file(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

def text_split(data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")
    return embeddings

def get_stock_info(symbol):
    """Get stock information and recent price data"""
    try:
        stock = yf.Ticker(f"{symbol}.NS")  # .NS for NSE stocks
        info = stock.info
        current_price = info.get('currentPrice', 'N/A')
        prev_close = info.get('previousClose', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        
        return {
            'name': info.get('longName', 'N/A'),
            'current_price': current_price,
            'prev_close': prev_close,
            'day_high': day_high,
            'day_low': day_low,
            'currency': 'INR'
        }
    except:
        return None

def get_mutual_fund_info(symbol):
    """Get mutual fund information"""
    try:
        mf = yf.Ticker(symbol)
        info = mf.info
        nav = info.get('regularMarketPrice', 'N/A')
        
        return {
            'name': info.get('longName', 'N/A'),
            'nav': nav,
            'category': info.get('category', 'N/A')
        }
    except:
        return None



