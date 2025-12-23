# ============================================================
# rag_server.py (LangChain RAG Refactor)
#
# 이 파일의 역할
# ------------------------------------------------------------
# 1) FastAPI 서버를 띄운다
# 2) Chroma 벡터DB를 로드한다 (이미 저장된 벡터 사용)
# 3) 질문이 오면:
#    - 벡터DB에서 관련 문서를 검색(Retriever)
#    - 검색된 문서를 CONTEXT로 만들어
#    - LLM(OpenAI)에게 답변을 생성하게 한다
#
# 실행 방법:
#   uvicorn rag_server:app --reload
# ============================================================

"""
서버 시작
 → 벡터DB 로드
   → /chat 요청
     → 문서 검색
       → 검색 결과를 CONTEXT로
         → LLM이 답변 생성
"""

# ----------------------------
# FastAPI 관련
# ----------------------------
# FastAPI: Python으로 API 서버를 만들기 위한 프레임워크
from fastapi import FastAPI

# Pydantic: API 요청(JSON)을 파이썬 객체로 변환 + 검증
from pydantic import BaseModel


# ----------------------------
# LangChain 관련 컴포넌트
# ----------------------------

# OpenAIEmbeddings:
# - 텍스트를 "벡터(숫자 배열)"로 바꿔주는 역할
# ChatOpenAI:
# - 실제 답변을 생성하는 LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Chroma:
# - 벡터DB (로컬 파일 기반)
from langchain_chroma import Chroma

# ChatPromptTemplate:
# - LLM에게 보낼 프롬프트 틀(template)
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser:
# - LLM의 응답을 문자열(str)로 깔끔하게 뽑아주는 파서
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# 프로젝트 설정값
# ----------------------------
# config.py에 모아둔 설정들
from config import (
    OPENAI_API_KEY,      # 테스트용 API 키
    EMBED_MODEL,         # 임베딩 모델 이름
    CHAT_MODEL,          # 답변 생성 모델 이름
    CHROMA_DIR,          # 벡터DB 저장 경로
    COLLECTION_NAME,     # 벡터DB 컬렉션 이름
    TOP_K,               # 검색할 문서 개수
)


# ============================================================
# FastAPI 앱 생성
# ============================================================
# app 객체가 "서버 그 자체"라고 보면 된다
app = FastAPI(title="LangChain RAG Server")


# ============================================================
# LangChain 핵심 구성요소들
# (서버 시작 시 한 번만 생성됨)
# ============================================================

# ----------------------------
# 1) 임베딩 객체
# ----------------------------
# 문서를 벡터로 바꿀 때 사용
# ingest 단계에서 쓴 모델과 반드시 같아야 한다
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
)


# ----------------------------
# 2) 벡터DB 로드
# ----------------------------
# 이미 ingest_langchain.py에서 저장한 벡터DB를 불러온다
vector_db = Chroma(
    collection_name=COLLECTION_NAME,   # 같은 컬렉션 이름
    embedding_function=embeddings,     # 같은 임베딩 방식
    persist_directory=CHROMA_DIR,      # 같은 저장 경로
)


# ----------------------------
# 3) Retriever 생성
# ----------------------------
# Retriever = "질문 → 관련 문서 검색" 담당
retriever = vector_db.as_retriever(
    search_kwargs={"k": TOP_K}          # 상위 k개 문서 검색
)


# ----------------------------
# 4) LLM 생성
# ----------------------------
# 실제 답변을 만들어주는 AI
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,   # 낮을수록 답변이 보수적/안정적
)


# ----------------------------
# 5) 프롬프트 템플릿
# ----------------------------
# LLM에게 "어떤 규칙으로 답변하라"라고 알려주는 부분
prompt = ChatPromptTemplate.from_template(
    """
너는 RAG 기반 QA 챗봇이다.
아래 CONTEXT에 있는 정보만 사용해서 답변해라.
모르면 "문서에서 근거를 찾지 못했습니다."라고 말해라.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
"""
)


# ----------------------------
# 6) 출력 파서
# ----------------------------
# LLM 응답을 문자열로 변환
output_parser = StrOutputParser()


# ============================================================
# Helper 함수
# ============================================================
def format_docs(docs):
    """
    검색된 Document 리스트를
    하나의 문자열(context)로 합치는 함수

    docs:
        [Document, Document, ...]
    """
    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# API 요청 데이터 구조
# ============================================================
class ChatRequest(BaseModel):
    """
    /chat API로 들어오는 JSON 형태

    {
      "question": "질문 내용"
    }
    """
    question: str


# ============================================================
# API 엔드포인트
# ============================================================
@app.post("/chat")
def chat(req: ChatRequest):
    """
    이 함수는:
    POST /chat 요청이 올 때마다 실행된다
    """

    # ----------------------------
    # 1) 문서 검색 (Retrieval)
    # ----------------------------
    # 질문을 기준으로 벡터DB에서 관련 문서 검색
    docs = retriever.invoke(req.question)


    # ----------------------------
    # 2) LangChain 체인 구성
    # ----------------------------
    # 체인의 흐름:
    #   context, question 생성
    #     → prompt에 삽입
    #       → LLM 호출
    #         → 문자열로 변환
    chain = (
        {
            "context": lambda _: format_docs(docs),
            "question": lambda _: req.question,
        }
        | prompt
        | llm
        | output_parser
    )


    # ----------------------------
    # 3) 체인 실행
    # ----------------------------
    answer = chain.invoke({})


    # ----------------------------
    # 4) 결과 반환 (JSON)
    # ----------------------------
    return {
        "ok": True,
        "question": req.question,
        "answer": answer,
        "sources": [
            d.metadata.get("source") for d in docs
        ],
    }
