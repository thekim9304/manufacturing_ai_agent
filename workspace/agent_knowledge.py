import json
import os
from opensearchpy import OpenSearch, helpers
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 환경 변수 로드 (GROQ_API_KEY 등)
load_dotenv()

class BatteryKnowledgeAgent:
    def __init__(self):
        # 1. OpenSearch 연결 설정
        self.client = OpenSearch(
            hosts=[{'host': 'opensearch', 'port': 9200}],
            http_auth=('admin', 'admin'),  # OpenSearch 기본 계정
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = "battery_manual_index"
        
        # 2. Groq LLM 설정 (Llama 3 70B - 추론 최적화 모델)
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,  # 제조 데이터의 일관성을 위해 0 설정
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def create_and_index(self, json_path='battery_knowledge.json'):
        """[색인 단계] JSON 지식을 OpenSearch에 저장 (BM25 자동 적용)"""
        # 인덱스가 이미 있다면 삭제 후 재생성 (실습 초기화용)
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        
        # 인덱스 설정 (한글 분석기가 없다면 기본 standard 사용, BM25는 기본값)
        settings = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=settings)

        # 데이터 로드 및 벌크 인덱싱
        with open(json_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "page": doc['page'],
                    "content": doc['content']
                }
            }
            for doc in docs
        ]
        
        helpers.bulk(self.client, actions)
        print(f"✅ 색인 완료: {len(docs)}개의 지식 섹션이 OpenSearch에 저장되었습니다.")

    def search_bm25(self, query, top_k=2):
        """[검색 단계] BM25 알고리즘을 사용한 키워드 검색"""
        query_body = {
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "or"  # 키워드 중 하나라도 포함되면 매칭
                    }
                }
            }
        }
        response = self.client.search(index=self.index_name, body=query_body)
        
        # 검색된 원문 리스트 반환
        retrieved_docs = [hit['_source']['content'] for hit in response['hits']['hits']]
        return retrieved_docs

    def generate_answer(self, query):
        """[생성 단계] 검색된 지식을 바탕으로 Groq LLM 응답 생성 (RAG)"""
        # 1. 관련 지식 검색
        context_list = self.search_bm25(query)
        print(context_list)
        context = "\n\n".join(context_list)
        
        if not context:
            return "가이드북에서 관련 내용을 찾을 수 없습니다."

        # 2. 시스템 프롬프트 구성 (전문가 페르소나 주입)
        prompt = f"""
        당신은 현대자동차 배터리 품질 관리 AI 전문가입니다. 
        제공된 [가이드북 내용]만을 근거로 삼아 사용자의 질문에 답변하세요. 
        가이드북에 없는 내용은 추측하지 마세요.

        [가이드북 내용]:
        {context}

        [사용자 질문]:
        {query}
        
        답변 (기술적으로 구체적이고 전문적인 어조로):"""

        # 3. LLM 호출
        response = self.llm.invoke(prompt)
        return response.content