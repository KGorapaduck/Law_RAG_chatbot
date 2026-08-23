# build_vector_db.py
import os
import glob
import json
import re
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_law_db_bge"

def parse_law_filename(filename):
    """파일명에서 상위법령명(parent_law) 및 세부구분(law_category) 추출"""
    base_name = os.path.basename(filename)
    
    parent_law = "알수없음"
    if "개인정보 보호법" in base_name or "개인정보보호법" in base_name:
        parent_law = "개인정보 보호법"
    elif "근로기준법" in base_name:
        parent_law = "근로기준법"
    elif "도로교통법" in base_name:
        parent_law = "도로교통법"
    elif "전자상거래" in base_name:
        parent_law = "전자상거래 등에서의 소비자보호에 관한 법률"
    elif "주택임대차" in base_name:
        parent_law = "주택임대차보호법"
        
    law_category = "법"
    if "시행규칙" in base_name:
        law_category = "시행규칙"
    elif "시행령" in base_name:
        law_category = "시행령"
    elif "(법률)" in base_name or "법률" in base_name:
        law_category = "법"
        
    return parent_law, law_category

def load_all_documents():
    json_files = glob.glob(os.path.join(DATA_DIR, "*_result.json"))
    documents = []
    
    print(f"[*] Total {len(json_files)} JSON files found. Processing...")
    
    for file_path in json_files:
        parent_law, law_category = parse_law_filename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            title = item.get("title", "").strip()
            
            # 조항 번호 파싱 (예: "제10조(목적)" -> 10, "제30조의2" -> 30)
            match = re.search(r"제(\d+)조", title)
            article_no = int(match.group(1)) if match else 0
            
            # 본문 및 세부 항목(items) 결합
            contents_list = item.get("contents", [])
            body_parts = []
            for c in contents_list:
                t = c.get("text", "").strip()
                if t:
                    body_parts.append(t)
                for itm in c.get("items", []):
                    itm_str = str(itm).strip()
                    if itm_str:
                        body_parts.append(itm_str)
                        
            body_text = "\n".join(body_parts)
            full_content = f"{parent_law} {law_category} {title}\n{body_text}".strip()
            
            metadata = {
                "parent_law": parent_law,
                "law_category": law_category,
                "article_no": article_no,
                "title": title
            }
            
            documents.append(Document(page_content=full_content, metadata=metadata))
            
    print(f"[+] Total {len(documents)} Document objects successfully created.")
    return documents

def build_db():
    docs = load_all_documents()
    
    print("[*] Loading BGE-M3 Embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 기존 DB 디렉토리가 있다면 초기화 후 재구축
    if os.path.exists(CHROMA_DIR):
        print(f"[*] Removing old DB directory ({CHROMA_DIR})...")
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception as e:
            print(f"[*] Directory locked by active process ({e}), clearing collection via Chroma API...")
            try:
                temp_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                temp_db.delete_collection()
            except Exception:
                pass
        
    print(f"[*] Rebuilding Chroma Vector DB at {CHROMA_DIR}...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"[+] Successfully built Vector DB with metadata at {CHROMA_DIR}!")

if __name__ == "__main__":
    build_db()
