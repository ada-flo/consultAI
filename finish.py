# openai GPT 모델 모듈 불러오기 
import openai
from openai.types import ChatModel
from openai.types.chat import ChatCompletion

## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import requests  # For making API calls
import pandas as pd  # For data manipulation
import plotly.express as px  # For data visualization
import re
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz
import re

## 환경변수 불러오기
from dotenv import load_dotenv,dotenv_values
load_dotenv()


############################### 1단계 : json 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: JSON 데이터를 Document로 변환
def json_to_documents(json_data: dict) -> List[Document]:
    documents = []
    # Assuming the JSON is structured with data you want to convert into documents
    if isinstance(json_data, list):
        for item in json_data:
            if 'content' in item:
                doc = Document(page_content=item['content'], metadata={"source": "API"})
                documents.append(doc)
    else:
        # Handling a single JSON object if it's not an array
        for key, value in json_data.items():
            if isinstance(value, str):
                doc = Document(page_content=value, metadata={"key": key, "source": "API"})
                documents.append(doc)
    return documents

## 2: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 3: Document를 벡터DB로 저장
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")



############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################


## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    ## 사용자 질문을 기반으로 관련문서 3개 검색 
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs

def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()

## Use GPT to extract the company name
def extract_company_name(user_question):
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Replace with your OpenAI API key
    
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Extract a valid company ticker in English from this question: '{user_question}' State only the ticker"}
        ],
        max_tokens=50,
        temperature=0
    )
    
    company_name = response.choices[0].message.content.strip()
    return company_name

############################### Alpha Vantage API 관련 함수 ##########################

def get_alpha_vantage_data(symbol: str, api_key: str) -> pd.DataFrame:
    """Fetch daily time series data for a given company symbol from Alpha Vantage API."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Parse data into DataFrame
    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.apply(pd.to_numeric)
        return df
    else:
        st.error("Failed to fetch data or invalid API response.")
        return pd.DataFrame()

def get_company_overview(symbol: str, api_key: str) -> dict:
    """Fetch company overview data for a given company symbol from Alpha Vantage API."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    # Check if the response contains valid data
    if data and "Symbol" in data:
        return data  # Return the raw JSON data as a dictionary
    else:
        st.error("Failed to fetch company overview or invalid API response.")
        return {}

def get_market_news(ticker: str, api_key: str) -> dict:
    """Fetch market news related to a given company ticker and topic from Alpha Vantage API."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    return data

def main():
    st.set_page_config("SoftlyAI 챗봇", layout="wide")
    
    st.header("SoftlyAI 챗봇")

    user_question = st.text_input("금융 관련 질문을 해주세요",
                                    placeholder="금융 관련 질문을 해주세요")

    if user_question:
        
        company_ticker = extract_company_name(user_question)  
        
        api_key = st.secrets["ALPHAVANTAGE_API_KEY"]  # Store API key in Streamlit secrets for security
        
        if company_ticker:
            with st.spinner("데이터를 불러오는 중..."):
                df = get_alpha_vantage_data(company_ticker, api_key)
                if not df.empty:
                    st.subheader(f"{company_ticker} 일일 주가 데이터")
                    st.dataframe(df.head())
                    # 시각화
                    fig = px.line(df, x=df.index, y="Close", title=f"{company_ticker} 주가 (종가)")
                    st.plotly_chart(fig)
        
        company_overview = get_company_overview(company_ticker, api_key)
        news_data = get_market_news(company_ticker, api_key)
        st.text(news_data)
        
        json_doc = json_to_documents(news_data)
        smaller_documents = chunk_documents(json_doc)
        save_to_vector_store(smaller_documents)
        
        response, context = process_question(user_question)
        st.write(response)
        i = 0 
        for document in context:
            with st.expander("관련 문서"):
                st.write(document.page_content)
                file_path = document.metadata.get('source', '')
                page_number = document.metadata.get('page', 0) + 1
                button_key =f"link_{file_path}_{page_number}_{i}"
                reference_button = st.button(f"🔎 {os.path.basename(file_path)} pg.{page_number}", key=button_key)
                if reference_button:
                    st.session_state.page_number = str(page_number)
                i = i + 1


if __name__ == "__main__":
    main()

