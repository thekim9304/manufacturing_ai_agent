import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import operator
from typing import TypedDict, Annotated, List, Dict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# 1. 환경 설정 및 세션 상태 로드
load_dotenv()

# --- [Phase 02] 로컬 에이전트 클래스 로드 ---
# 프로젝트 루트에 agent_data.py와 agent_knowledge.py가 있어야 합니다.
from agent_data import BatteryDataAgent
from agent_knowledge import BatteryKnowledgeAgent

# --- [Phase 03] LangGraph 상태 및 통합 에이전트 설정 ---

class AgentState(TypedDict):
    query: str
    db_results: str
    manual_results: str
    next_step: str
    final_answer: str
    # Annotated와 operator.add를 사용하여 여러 노드의 로그를 리스트에 누적합니다[cite: 40].
    logs: Annotated[list, operator.add] 

class BatteryOrchestrator:
    def __init__(self):
        # 최신 Llama-3.3 모델을 사용하여 추론 속도와 비용 효율성을 확보했습니다.
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.data_agent = BatteryDataAgent(
            db_path = "/home/jovyan/project/data/battery_factory.db",
            dict_path = "/home/jovyan/project/data/battery_semantic_dict.json"
        )
        self.knowledge_agent = BatteryKnowledgeAgent()

    def router(self, state: AgentState):
        """[Intent Classification] 질문의 의도를 분석하여 최적의 경로 선택 [cite: 66]"""
        prompt = f"""
        사용자의 질문을 분석하여 가장 적합한 처리 경로를 하나만 선택하세요.
        1. LLM: 단순 인사, 시스템 정체성 질문, 혹은 일반적인 상식 질문
        2. DATA: 배터리 팩 수치 조회, 평균/최댓값 계산 등 DB 조회가 필요한 경우
        3. KNOWLEDGE: 불량 원인, 조치 방법 등 기술 가이드북 지식이 필요한 경우
        4. BOTH: 수치 확인과 기술적 진단이 동시에 필요한 경우
        
        질문: {state['query']}
        결과(LLM/DATA/KNOWLEDGE/BOTH):"""
        
        decision = self.llm.invoke(prompt).content.strip().upper()
        
        log = {
            "node": "🎯 Router",
            "description": "질문 의도 분류 및 전문가 할당 [cite: 4, 66]",
            "prompt": prompt,
            "input": state['query'],
            "output": decision
        }
        return {"next_step": decision, "logs": [log]}

    def call_general_llm(self, state: AgentState):
        """[General AI] 외부 데이터 없이 LLM의 기본 지식으로 즉시 응답 [cite: 17]"""
        res = self.llm.invoke(state['query']).content
        log = {
            "node": "🤖 General LLM",
            "description": "일반 상식/인사 질문에 대한 직접 응답",
            "prompt": "Direct LLM Inference",
            "input": state['query'],
            "output": "응답 생성 완료"
        }
        return {"final_answer": res, "logs": [log]}

    def call_data_expert(self, state: AgentState):
        """[Data Expert] 1.1억 건 대규모 시계열 로그 분석 [cite: 6, 8, 171]"""
        res = self.data_agent.execute_and_analyze(state['query'])
        log = {
            "node": "📊 Data Expert",
            "description": "Text2SQL 기반 제조 데이터 수치 분석 [cite: 43]",
            "prompt": "Internal Text2SQL Generation",
            "input": state['query'],
            "output": "SQL 쿼리 실행 성공",
            "data": res
        }
        return {"db_results": res, "logs": [log]}

    def call_knowledge_expert(self, state: AgentState):
        """[Knowledge Expert] 가이드북 지식 검색 (Hybrid Search) [cite: 4, 56, 122]"""
        res = self.knowledge_agent.generate_answer(state['query'])
        log = {
            "node": "📚 Knowledge Expert",
            "description": "OpenSearch 기반 도메인 지식 추출 [cite: 16, 85]",
            "prompt": "RAG Context Retrieval",
            "input": state['query'],
            "output": "문서 검색 성공",
            "data": res
        }
        return {"manual_results": res, "logs": [log]}

    def grade_documents(self, state: AgentState):
        """[CRAG Grader] 검색 결과의 관련성을 스스로 채점하여 환각 방지 [cite: 115, 117]"""
        context = state.get('manual_results', '')
        query = state['query']
        prompt = f"문서 품질 평가 프롬프트... 질문: {query}"
        
        grade_res = self.llm.invoke(prompt).content.strip().lower()
        decision = "PROCESS" if 'yes' in grade_res else "REWRITE"
        
        log = {
            "node": "🔍 CRAG Grader",
            "description": "검색된 지식의 질문 적합성 검증 [cite: 115]",
            "prompt": prompt,
            "input": {"context_len": len(context)},
            "output": decision
        }
        return {"next_step": decision, "logs": [log]}

    def rewrite_query(self, state: AgentState):
        """[CRAG Rewriter] 검색 실패 시 쿼리 재구성을 통한 검색 품질 개선 [cite: 4, 115]"""
        prompt = f"질문 재구성 프롬프트... 원래 질문: {state['query']}"
        new_query = self.llm.invoke(prompt).content.strip()
        res = self.knowledge_agent.generate_answer(new_query)
        
        log = {
            "node": "🔄 CRAG Rewriter",
            "description": "Query Rewriting을 통한 재검색 실행 [cite: 116]",
            "prompt": prompt,
            "input": state['query'],
            "output": new_query
        }
        return {"query": new_query, "manual_results": res, "next_step": "PROCESS", "logs": [log]}

    def final_responder(self, state: AgentState):
        """[Synthesis] 데이터와 지식을 통합한 최종 리포트 작성 [cite: 3, 111]"""
        summary_prompt = f"종합 분석 프롬프트... 데이터: {state.get('db_results', '')}"
        response = self.llm.invoke(summary_prompt).content
        
        log = {
            "node": "🏁 Final Responder",
            "description": "멀티모달 분석 결과 종합 요약 [cite: 145]",
            "prompt": summary_prompt,
            "output": "최종 분석 보고서 생성 완료"
        }
        return {"final_answer": response, "logs": [log]}

# --- LangGraph 워크플로우 구성 및 컴파일 ---

orchestrator = BatteryOrchestrator()
workflow = StateGraph(AgentState)

# 1. 노드 등록
workflow.add_node("router", orchestrator.router)
workflow.add_node("general_llm", orchestrator.call_general_llm)
workflow.add_node("data_expert", orchestrator.call_data_expert)
workflow.add_node("knowledge_expert", orchestrator.call_knowledge_expert)
workflow.add_node("grader", orchestrator.grade_documents)
workflow.add_node("rewriter", orchestrator.rewrite_query)
workflow.add_node("final_responder", orchestrator.final_responder)

# 2. 엣지 연결 (조건부 분기 포함)
workflow.set_entry_point("router")

def route_logic(state):
    step = state["next_step"]
    if step == "LLM": return "general_llm"
    if step == "DATA": return "data_expert"
    if step == "KNOWLEDGE": return "knowledge_expert"
    return "data_expert" # BOTH 기본값

workflow.add_conditional_edges("router", route_logic, {
    "general_llm": "general_llm",
    "data_expert": "data_expert",
    "knowledge_expert": "knowledge_expert"
})

workflow.add_edge("general_llm", END)
workflow.add_edge("data_expert", "knowledge_expert")
workflow.add_edge("knowledge_expert", "grader")

workflow.add_conditional_edges("grader", lambda x: x["next_step"], {
    "PROCESS": "final_responder",
    "REWRITE": "rewriter"
})
workflow.add_edge("rewriter", "final_responder")
workflow.add_edge("final_responder", END)

app = workflow.compile()

# --- [UI] Streamlit 인터페이스 ---

st.set_page_config(page_title="HMG Battery QA Agent", layout="wide")
st.title("🔋 배터리팩 품질보증 지능형 관제 시스템")

# 세션 상태 초기화 (메시지 바구니 생성) [cite: 39]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문하세요 (예: 1000번 팩의 평균 전압은?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 추론 과정을 투명하게 공개하는 화이트박스 트레이스 컨테이너 [cite: 40]
        log_expander = st.expander("🛠️ 에이전트 상세 추론 과정 (White-box Trace)", expanded=False)
        
        with st.status("CRAG 기반 정밀 분석 중...", expanded=True) as status:
            input_data = {"query": prompt, "logs": []}
            final_ans = ""
            
            for output in app.stream(input_data):
                for node_name, node_state in output.items():
                    # 1. 사이드바에 실시간 진행 상황 표시
                    st.write(f"✅ `{node_name}` 노드 실행 완료")
                    
                    # 2. 상세 추론 로그 기록 (Expander 내부 시각화)
                    if "logs" in node_state:
                        for log in node_state["logs"]:
                            with log_expander:
                                st.subheader(f"{log['node']}")
                                st.caption(log.get('description', ''))
                                with st.container(border=True):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**📥 프롬프트/입력**")
                                        st.code(log.get('prompt', '') or log.get('input', ''), language="markdown")
                                    with col2:
                                        st.markdown("**📤 결과/판단**")
                                        st.info(log.get('output', ''))
                                        if 'data' in log:
                                            with st.expander("📦 상세 데이터 (Raw Data)"):
                                                st.json(log['data'])
                                st.divider()
                    
                    if node_name == "final_responder" or node_name == "general_llm":
                        final_ans = node_state.get("final_answer", "")
            
            status.update(label="분석 완료!", state="complete")
        
        st.markdown(final_ans)
        st.session_state.messages.append({"role": "assistant", "content": final_ans})