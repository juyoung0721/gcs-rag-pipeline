# gcs-rag-pipeline

Google Cloud Storage(GCS)에 업로드된 문서를 자동으로 파싱 및 전처리하여, RAG(Retrieval-Augmented Generation) 시스템에 연동하는 전체 파이프라인 예제입니다.

## 구성 폴더 및 설명

- **gcs-function/**  
  GCS 파일 업로드 이벤트를 트리거로 파일 파싱/전처리 및 색인(indexing)을 수행하는 Cloud Function 코드

- **rag_tool_api/**  
  전처리된 데이터 기반 RAG API 서버(FastAPI 기반). 문서 검색 및 LLM 기반 질의응답 기능 포함

## Cloud Function (gcs-function/)

### 1. 개요
- 주요 Google Cloud 서비스:  
  - **Google Cloud Storage (GCS)**: 파일 저장 및 트리거
  - **Cloud Functions (2nd Gen, Python 3.13)**: 이벤트 기반 서버리스 처리
  - **Vertex AI Search (Discovery Engine)**: 검색 및 RAG 인덱싱
  - **Secret Manager**: 민감 정보 관리
  - **IAM**: 권한 관리
  - **Cloud Build**: 컨테이너 이미지 빌드

### 2. 폴더/파일 구조

```

gcs-function/
├── indexing.py      # 청킹 및 색인
├── main.py          # Cloud Function 엔트리포인트
├── parsers.py       # 파일 파싱
├── processing.py    # 전처리
└── requirements.txt # 의존성

```

### 3. 동작 흐름

```

GCS 파일 업로드
↓
Cloud Function(main.py) 실행
↓
파일 파싱/전처리 (processing.py)
↓
청킹 및 Vertex AI Search 인덱싱 (indexing.py)
↓
Vertex AI Search Datastore 저장

````

### 4. 환경/배포

- 런타임: Python 3.13, 1024 MB
- 필수 환경 변수: `PROJECT_ID`, `DATASTORE_ID`
- Secret: `OPENAI_API_KEY` (Secret Manager)
- 주요 IAM 권한:  
  - `roles/discoveryengine.admin`
  - `roles/secretmanager.secretAccessor`
  - `roles/storage.objectAdmin`
  - 등

### 배포 예시

```bash
gcloud functions deploy <cloud_function_name> \
  --gen2 --runtime python313 --region us-central1 \
  --memory=1024MiB --timeout=540s --source=. \
  --entry-point gcs_cloudevent \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=<bucket_name>" \
  --service-account <service-account> \
  --allow-unauthenticated \
  --set-env-vars=PROJECT_ID=<project id>,DATASTORE_ID=<datastore id> \
  --set-secrets=OPENAI_API_KEY=<SecretManagerURL>
````

## RAG API (rag_tool_api/)


### 1. 개요

- **목적:** 검색·RAG(검색 증강 생성) 답변을 제공하는 REST API.  
  (웹 챗봇/내부 서비스가 호출 → 검색(Retrieval) → LLM 프롬프트 생성 → 최종 답변 반환)
- **주요 서비스:**  
  - **Cloud Run** (또는 GKE): FastAPI 기반 컨테이너 실행  
  - **Vertex AI Search (Discovery Engine):** 색인된 문서 검색  
  - **Secret Manager:** OpenAI API Key 등 관리  
  - **IAM:** 서비스 계정 권한

---

### 2. 파일 구조

```

rag\_tool\_api/
├── Dockerfile
├── .dockerignore
├── main.py             # FastAPI 엔트리포인트/라우터
├── retrieval.py        # Vertex AI Search 검색 로직
├── llm\_answer.py       # 프롬프트 생성 및 LLM 호출
├── excel\_analysis.py   # 엑셀 시트 통계/요약
├── requirements.txt    # Python 의존성

```

---

### 3. 동작 흐름

```

Client → FastAPI(main.py) → retrieval.py(검색) / excel\_analysis.py(분석)
→ llm\_answer.py(프롬프트 생성+LLM) → JSON 응답 반환

````

---

### 4. 환경 구성 & 배포

- **런타임:** Python 3.13 (Uvicorn)
- **환경 변수:**  
    - `PROJECT_ID`: GCP 프로젝트 ID  
    - `ENGINE_ID`: Vertex AI Search 엔진 ID
- **Secret Manager:** `OPENAI_API_KEY`
- **IAM 권한:**  
    - `roles/discoveryengine.documentAdmin` 또는 `discoveryengine.editor`  
    - `roles/secretmanager.secretAccessor`
    - `roles/storage.objectViewer` (옵션)
- **Cloud Run 배포 예시**
    ```bash
    gcloud run deploy rag-tool-api \
      --source=. \
      --region=us-central1 \
      --platform=managed \
      --service-account <service-account> \
      --allow-unauthenticated \
      --memory=512Mi \
      --timeout=540 \
      --set-env-vars=PROJECT_ID=<project id>,ENGINE_ID=<engine id> \
      --set-secrets=OPENAI_API_KEY=<SecretManagerURL>
    ```

---

### 5. `/retrieve` 엔드포인트

- **URL:** `GET /retrieve`
- **Query Parameters**
    | 이름         | 타입   | 필수 | 기본값 | 설명                   |
    |--------------|--------|------|--------|------------------------|
    | workspaceId  | string | ✅   | -      | 워크스페이스 식별자    |
    | query        | string | ✅   | -      | 사용자 검색어          |
    | k            | int    | ❌   | 5      | 반환할 검색 청크 개수  |
- **성공 응답 예시**
    ```json
    {
      "summary": "<LLM 생성 요약>",
      "chunks": [
        {
          "file_name": "price_list.pdf",
          "content": "청크 본문",
          "score": 0.945,
          "metadata": {
            "workspace_id": "ws_12345",
            "file_type": "pdf",
            "parsing_type": "pdf_text",
            "file_path": "gs://bucket/raw/price_list.pdf",
            "chunk_number": 12,
            "page_number": 10,
            "sheet_name": ""
          }
        }
      ]
    }
    ```
- **필드 설명**
    - `summary`: LLM이 생성한 요약
    - `chunks`: 검색된 청크 목록 (각종 메타데이터 포함)
