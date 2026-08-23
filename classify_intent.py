# classify_intent.py
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def analyze_intent(state: dict, llm):
    """대화 기록(History)을 참조하여 의도 분석 및 메타데이터 필터 추출"""
    tried = ", ".join(state.get("tried_keywords", []))
    if state.get("needed_article"):
        needed = state["needed_article"]
        # 조항 번호 숫자만 추출 (예: "제10조" -> 10)
        match = re.search(r"(\d+)", needed)
        art_no = int(match.group(1)) if match else 0
        
        return {
            "keywords": needed,  # "근로기준법 시행령 제3조" 등을 키워드로 사용
            "intent": "legal",
            "needed_article": None  # 요청 접수 완료 (초기화)
        }
    
    system_msg = f"""당신은 법률 질문 분석 전문가입니다. 질문은 보통 상위 법령에 대해 물어볼 것입니다. 질문에 상위 법령이 들어갈 수 있는지 생각하고, 질문에서 상위 법령명과 세부 구분을 추출하세요.
    
    [상위 법령(parent_law)]
    개인정보 보호법, 근로기준법, 도로교통법, 전자상거래 등에서의 소비자보호에 관한 법률, 주택임대차보호법
    
    [세부 구분(law_category)]
    법, 시행령, 시행규칙, 지침

    [응답 형식]
    - 법률 질문: 'legal | 상위법령명 | 세부구분 | 조항번호(숫자만, 없으면 0) | 검색키워드'
      * 세부구분이 모호하면 '전체'로 표시하세요.
      * [재검색 루프 대응]: 이미 시도했지만 실패한 키워드들 [{tried}]은 피해서 더 정확한 법적 용어로 검색키워드를 만드세요.
      (예: "야간 수당을 달라고 했더니, 사장님이 '우리는 4명뿐이라 안 줘도 된다'고 하시네요. 알바생까지 합치면 6명인데 누구 말이 맞나요?" -> "legal | 근로기준법 | 전체 | 0 | 근로기준법 야간수당 및 상시근로자 수", "[상위법령명]이 뭐야" 혹은 "[상위법령명]에 대해 알려줘" -> "legal | [상위법령명] | 법 | 1 | [상위법령명]")
    - 일반 대화: 'general'
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # History를 포함하여 LLM 호출
    res = llm.invoke(prompt.format(history=state["history"], question=state["question"]))
    raw = res.content.strip()
    
    if "general" in raw or "|" not in raw:
        return {"intent": "general", "keywords": state["question"], "filter": None}

    parts = [i.strip() for i in raw.split("|")]
    p_law = parts[1] if len(parts) > 1 else "알수없음"
    l_cat = parts[2] if len(parts) > 2 else "전체"
    f_art = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    new_keyword = parts[4] if len(parts) > 4 else state["question"]

    # 메타데이터 필터 구성
    meta_filter = {}
    if p_law != "알수없음": meta_filter["parent_law"] = p_law
    if l_cat in ["법", "시행령", "시행규칙", "지침", "규정"]: meta_filter["law_category"] = l_cat
    if f_art > 0: meta_filter["article_no"] = f_art
    
    final_filter = {"$and": [{k: v} for k, v in meta_filter.items()]} if len(meta_filter) > 1 else meta_filter

    return {
        "intent": "legal",
        "filter": final_filter,
        "keywords": new_keyword,
        "tried_keywords": state.get("tried_keywords", []) + [new_keyword]
    }
