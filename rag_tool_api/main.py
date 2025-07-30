# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from retrieval import build_serving_config
from llm_answer import llm_answer
import os

# ── 환경변수 로드 ─────────────────────────────────────────────
PROJECT_ID = os.environ["PROJECT_ID"]
ENGINE_ID  = os.environ["ENGINE_ID"]

# ── FastAPI 앱 ───────────────────────────────────────────────
app = FastAPI(title="RAG Retrieve API")

# ── 요청/응답 모델 ───────────────────────────────────────────
class RetrieveRequest(BaseModel):
    workspaceId: str
    query: str
    k: int = 5

class RetrieveResponse(BaseModel):
    summary: str
    chunks: List[Dict]

@app.get("/retrieve", response_model=RetrieveResponse)
def retrieve(workspaceId: str, query: str, k: int = 5):
    """
    쿼리 기반 RAG 검색 + 요약 생성 API

    - 지정된 workspace 내에서 Vertex AI Search로 semantic chunk 검색
    - 검색 결과를 기반으로 OpenAI LLM이 요약 생성

    Args:
        workspaceId (str): 검색 대상 워크스페이스 ID
        query (str): 사용자 질문 또는 질의
        k (int): 검색 결과 수 (default: 5)

    Returns:
        RetrieveResponse: 요약 및 사용된 참조 chunk 목록
    """
    serving_config = build_serving_config(
        engine_id=ENGINE_ID,
        project_id=PROJECT_ID,
        location="global"
    )
    summary, chunks = llm_answer(
        mode="chunk",
        workspace_id=workspaceId,
        search_query=query,
        page_size=k,
        model="gpt-4.1",
        serving_config=serving_config
    )
    
    return RetrieveResponse(summary=summary, chunks=chunks)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
