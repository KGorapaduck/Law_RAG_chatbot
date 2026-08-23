# graph.py
from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from classify_intent import analyze_intent
from generate_answer import retrieve_law, generate_answer

# 1. LangGraph 상태(State) 정의
class GraphState(TypedDict):
    question: str
    history: List[BaseMessage]
    intent: str
    filter: Optional[dict]
    keywords: str
    tried_keywords: List[str]
    context: str
    answer: str
    retry_count: int
    needed_article: Optional[str]
    used_laws: str

# 2. 재검색 여부 분기 함수
def decide_next(state: GraphState):
    if state.get("intent") == "general":
        return "end"
    if "IDK" in state.get("answer", "") and state.get("retry_count", 0) < 2:
        return "retry"
    return "end"

# 3. 그래프 조립
def create_law_graph(vectordb, llm):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("analyze", lambda state: analyze_intent(state, llm))
    workflow.add_node("retrieve", lambda state: retrieve_law(state, vectordb))
    workflow.add_node("generate", lambda state: generate_answer(state, llm))
    
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges("generate", decide_next, {"retry": "analyze", "end": END})
    
    return workflow.compile()