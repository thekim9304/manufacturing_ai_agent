import sqlite3
import json
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class BatteryDataAgent:
    def __init__(self, db_path='battery_factory.db', dict_path='battery_semantic_dict.json'):
        self.db_path = db_path
        # 1. 시맨틱 사전 로드 (M01CV01 -> 1번 모듈 1번 셀 전압)
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.semantic_dict = json.load(f)
        
        # 2. Groq LLM 설정 (최신 모델 사용)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def get_db_schema(self):
        """DB 테이블 구조 추출 (에이전트에게 전달용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(battery_logs);")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        return columns

    def generate_sql(self, user_query):
        """자연어를 SQL로 변환"""
        columns = self.get_db_schema()
        
        # LLM에게 전달할 프롬프트 (사전 정보 포함)
        prompt = f"""
        당신은 SQLite 전문가입니다. 아래 정보를 바탕으로 사용자의 질문을 해결하는 SQL 쿼리만 작성하세요.
        
        [테이블 이름]: battery_logs
        [전체 컬럼명]: {', '.join(columns[:20])}... (총 {len(columns)}개)
        [주요 매핑 사전]:
        {json.dumps(dict(list(self.semantic_dict.items())[:10]), ensure_ascii=False)}
        
        [규칙]:
        1. 시계열 데이터이므로 필요시 'Time' 컬럼을 정렬에 사용하세요.
        2. 'SerialNumber'는 배터리팩 아이디입니다.
        3. 결과는 오직 SQL 쿼리문만 출력하세요. 설명은 필요 없습니다.

        [사용자 질문]: {user_query}
        
        SQL:"""
        
        response = self.llm.invoke(prompt)
        return response.content.strip().replace("```sql", "").replace("```", "")

    def execute_and_analyze(self, user_query):
        """SQL 생성 -> 실행 -> 결과 요약"""
        sql = self.generate_sql(user_query)
        print(f"📡 생성된 SQL: {sql}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            # 데이터를 Pandas로 읽어오면 분석이 더 쉽습니다.
            import pandas as pd
            df_result = pd.read_sql_query(sql, conn)
            conn.close()
            
            # 결과가 너무 크면 요약해서 LLM에게 전달
            result_str = df_result.to_string() if len(df_result) < 10 else f"총 {len(df_result)}건의 데이터가 검색되었습니다."
            
            # 3. 최종 결과 해석
            analysis_prompt = f"""
            질문: {user_query}
            조회 결과: {result_str}
            
            위 데이터를 바탕으로 사용자에게 친절하고 전문적으로 답변하세요.
            """
            final_response = self.llm.invoke(analysis_prompt)
            return final_response.content

        except Exception as e:
            return f"❌ 데이터 조회 중 오류 발생: {e}"