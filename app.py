# app.py
import re
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from data_base import load_resources
from graph import create_law_graph

# 1. 페이지 설정
st.set_page_config(page_title="지능형 법률 비서", page_icon="⚖️", layout="wide")

# 2. 세션 상태 초기화
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"session_1": "새 대화 1"}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "session_1"
if "session_messages" not in st.session_state:
    st.session_state.session_messages = {"session_1": []}
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

# 3. 리소스 및 그래프 로드 (캐싱 적용)
@st.cache_resource
def init_app():
    vectordb, llm = load_resources()
    graph_app = create_law_graph(vectordb, llm)
    return vectordb, llm, graph_app

vectordb, llm, graph_app = init_app()

# 현재 세션 대화 메시지
cur_messages = st.session_state.session_messages[st.session_state.current_session_id]

# 4. 사이드바 세션 관리 UI
with st.sidebar:
    st.subheader("📂 대화 목록")
    if st.button("+ 새 채팅 시작", use_container_width=True):
        new_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.chat_sessions[new_id] = f"새 대화 {len(st.session_state.chat_sessions) + 1}"
        st.session_state.session_messages[new_id] = []
        st.session_state.current_session_id = new_id
        st.rerun()
    
    for sid, name in list(st.session_state.chat_sessions.items()):
        col_n, col_d = st.columns([0.8, 0.2])
        is_cur = (sid == st.session_state.current_session_id)
        
        if st.session_state.get(f"edit_{sid}", False):
            new_name = col_n.text_input("수정", value=name, key=f"in_{sid}", label_visibility="collapsed")
            if col_d.button("✅", key=f"sv_{sid}"):
                st.session_state.chat_sessions[sid] = new_name
                st.session_state[f"edit_{sid}"] = False
                st.rerun()
        else:
            btn_label = f"> {name}" if is_cur else name
            if col_n.button(btn_label, key=f"sel_{sid}", use_container_width=True):
                if is_cur:
                    st.session_state[f"edit_{sid}"] = True
                else:
                    st.session_state.current_session_id = sid
                st.rerun()
            if col_d.button("🗑️", key=f"del_{sid}"):
                if len(st.session_state.chat_sessions) > 1:
                    del st.session_state.chat_sessions[sid]
                    del st.session_state.session_messages[sid]
                    if is_cur:
                        st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
                    st.rerun()

# 5. 메인 채팅 화면 UI
st.title("⚖️ 생활 법률 질문 AI 비서")

# 대화 내용 출력 (인용 법령 expander 포함)
for msg in cur_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)
        if role == "assistant" and "used_laws" in msg.additional_kwargs:
            if msg.additional_kwargs["used_laws"]:
                with st.expander("⚖️ 검색된 법령 원문 전체 보기"):
                    st.markdown(msg.additional_kwargs["used_laws"])

# 채팅 입력창
if user_input := st.chat_input("질문을 입력하세요...", disabled=st.session_state.is_thinking):
    cur_messages.append(HumanMessage(content=user_input))
    st.session_state.is_thinking = True
    st.rerun()

# 답변 생성 프로세스
if st.session_state.is_thinking:
    with st.chat_message("assistant"):
        with st.status("⚖️ 법령을 검토하고 답변을 생성 중입니다...", expanded=True) as status:
            last_question = cur_messages[-1].content
            chat_history = cur_messages[:-1]
            
            final_res = {
                "question": last_question, 
                "history": chat_history, 
                "tried_keywords": [], 
                "retry_count": 0, 
                "context": "",
                "needed_article": None, 
                "used_laws": "", 
                "answer": ""
            }
            
            for i in range(3):
                if i > 0:
                    status.update(label=f"🔎 정보가 부족하여 재검색 중입니다... ({i}/3)", state="running")
                
                final_res = graph_app.invoke(final_res)
                
                if not final_res.get("needed_article") and "IDK" not in final_res.get("answer", ""):
                    break
                
                final_res["retry_count"] += 1
                st.toast(f"🔄 추가 근거를 찾는 중... ({i+1}/3)")
            
            status.update(label="✅ 검토 완료", state="complete", expanded=False)

        raw_answer = final_res.get("answer", "")
        raw_context = final_res.get("context", "") 
        
        # 참고 번호 정규표현식 파싱
        used_indices = []
        match = re.search(r"(?:USED_LAWS|참고\s*번호|인용\s*번호):\s*([\d,\s]+)", raw_answer)
        if match:
            used_indices = [idx.strip() for idx in match.group(1).split(",")]

        # 실제로 인용된 법령 본문만 필터링
        filtered_laws = []
        if used_indices and raw_context:
            law_blocks = re.split(r"(\[\d+\])", raw_context)
            for i in range(1, len(law_blocks), 2):
                block_num = law_blocks[i].replace("[", "").replace("]", "")
                if block_num in used_indices:
                    law_content = law_blocks[i+1].strip()
                    filtered_laws.append(law_content)

        law_context_to_save = "\n\n---\n\n".join(filtered_laws) if filtered_laws else raw_context
        
        # 시스템용 태그 제거 및 답변 정제
        display_answer = raw_answer.replace("IDK", "").replace("(추가 조항 검색 중...)", "").strip()
        clean_answer = re.sub(r"(USED_LAWS|참고\s*번호|인용\s*번호):\s*[\d,\s]+", "", display_answer).strip()
        
        st.write(clean_answer)

        # AI 메시지 저장
        ai_msg = AIMessage(
            content=clean_answer, 
            additional_kwargs={"used_laws": law_context_to_save}
        )
        cur_messages.append(ai_msg)
        
        if law_context_to_save:
            with st.expander("⚖️ 검색된 법령 원문 전체 보기" if not filtered_laws else "⚖️ 답변에 인용된 법령 원문"):
                st.markdown(law_context_to_save)
    
    st.session_state.is_thinking = False
    st.rerun()