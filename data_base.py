# 벡터DB,llm
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)
CHROMA_DIR = "./chroma_law_db_bge"

def load_resources():
    print("🔄 리소스 로딩 중 (BGE-M3 모델 및 벡터 DB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    return vectordb, llm