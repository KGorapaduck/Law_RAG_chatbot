# generate_answer.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def retrieve_law(state: dict, vectordb):
    """법령 문서 검색 (메타데이터 필터링 및 Self-RAG 재검색 지원)"""
    if state.get("intent") != "legal": 
        return {"context": state.get("context", "")}
    
    # 검색 개수를 늘려(k=6) 충분한 정보를 가져옵니다.
    search_kwargs = {"k": 6}
    if state.get("filter"): 
        search_kwargs["filter"] = state["filter"]
    
    # 새로운 조항 검색 실행
    docs = vectordb.as_retriever(search_kwargs=search_kwargs).invoke(state["keywords"])
    
    # 검색된 문서들에 번호를 매겨서 '진짜 본문'임을 명시합니다.
    formatted_docs = []
    for i, d in enumerate(docs):
        content = d.page_content.strip()
        formatted_docs.append(f"[{i+1}] {content}")
    
    new_context = "\n\n".join(formatted_docs)
    
    # 기존 context가 있다면 유지하고 뒤에 추가 (Self-RAG 루프 대응)
    old_context = state.get("context", "")
    if old_context:
        combined_context = f"{old_context}\n\n--- [추가 재검색 결과] ---\n{new_context}"
    else:
        combined_context = f"--- [검색된 법령 원문] ---\n{new_context}"
    
    return {"context": combined_context.strip()}

def generate_answer(state: dict, llm):
    """법령 기반 답변 생성 (IDK 체크 및 인용 번호 포함)"""
    if state.get("intent") == "general":
        system_msg = """당신은 대한민국 '생활 법률(개인정보 보호법, 근로기준법, 도로교통법, 전자상거래법, 주택임대차 보호법 및 그 시행령과 시행 규칙)' 전문 AI 비서입니다. 

        [응대 원칙]
        1. 정체성 유지: 사용자의 일상적인 인사나 일상적인 대화에는 친절하게 응답하세요.
        2. 답변 거부 기준: 아래의 경우 "제가 도움을 드릴 수 있는 범위를 벗어난 질문입니다"라고 정중히 답변하세요.
           - 생활 법률과 관련 없는 다른 법률(형법, 민법, 상법 등)에 관한 구체적인 상담
           - 불법적인 행위를 조장하거나 편법을 묻는 질문
           - 정치, 종교, 연예 등 법률 비서의 역할과 무관한 의견 요구
           - 법률과 관계없는 전문적인 지식(예: 요리법, 과학 지식 등)
        3. 맥락 파악: 사용자가 "이전에 말한 것"에 대해 묻는다면 대화 기록(History)을 참고하여 연결성 있게 대답하세요.
        4. 전문성 안내: 답변 끝에는 가끔씩 사용자가 생활 법률에 대해 궁금한 점이 있다면 언제든 물어봐 달라는 안내를 덧붙이세요.
        5. 전문적인 지식이 필요하지 않은 질문이라면 간단하고 재치있게 답변하세요.

        ---
        [대화 예시]
        질문: "피자 만드는 법 알려줘"
        답변: "피자 레시피와 같은 전문적인 지식은 제가 알려드리기 어렵습니다! 저는 생활 법률(근로기준, 도로교통 등) 전문 비서이니, 혹시 법률적으로 궁금한 점이 생기시면 언제든 말씀해 주세요."

        질문: "살인죄 형량은 어떻게 돼?"
        답변: "죄송합니다. 저는 생활 법률(개인정보, 근로기준, 도로교통, 전자상거래, 주택임대차)을 중심으로 도움을 드리고 있어, 형법에 해당하는 살인죄 형량에 대해서는 답변을 드리기 어렵습니다. 관련된 전문 법률 상담을 이용하시길 권장드립니다."
        """
    else:
        system_msg = """당신은 대한민국 법률 전문가입니다. 제공된 [법령 정보]의 번호(예: [1], [3])를 인용하여 사용자의 질문에 신뢰성 있는 답변을 제공해야 합니다.

        [답변 가이드라인]
        1. 근거 중심 답변: 반드시 제공된 [법령 정보]만을 근거로 답하세요. 법령에 없는 내용을 추측하거나 임의로 판단하지 마세요.
        2. 조문 명시: 답변 내용 중 인용한 법령의 명칭과 조항 번호(예: 근로기준법 제0조 제0항)를 반드시 명시하세요.
        3. (중요)제공된 [법령 정보]만으로 답변이 불가능한 경우, 필요한 조항이 있다면 'NEED_MORE: 조항명' 형식으로 추가 조항명을 명시하고, 대답할 수 없다고 생각이 될 땐 'IDK'라는 단어를 사용하세요. 
        4. 만약 더 정확한 상담을 위해 필요한 추가 정보가 있다면 사용자에게 질문하세요.
        5. (중요)답변 끝에는 반드시 'USED_LAWS: 1, 3' 처럼 참고한 번호만 나열하세요.
        ---
        [법령 정보]
        {context}

        ---
        [답변 구조 예시]
        답변: (조항을 포함한 상세 설명)
        요약: (사용자가 이해하기 쉬운 요약 한 문장)
        USED_LAWS: 1, 2
        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg.format(context=state.get("context", ""))),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    res = llm.invoke(prompt.format(history=state.get("history", []), question=state["question"]))
    full_content = res.content

    if "NEED_MORE:" in full_content:
        needed = full_content.split("NEED_MORE:")[1].split("\n")[0].strip()
        return {"needed_article": needed, "answer": "IDK (추가 조항 검색 중...)", "context": state.get("context", "")}

    return {
        "answer": full_content.strip(),
        "context": state.get("context", ""),
        "intent": state.get("intent", "legal")
    }