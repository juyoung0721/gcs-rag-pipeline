import os
from typing import List, Dict, Any, Tuple
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf.json_format import MessageToDict


def build_serving_config(
    engine_id: str,
    project_id: str | None = None,
    location: str | None = None,
    serving_name: str = "default_config",
) -> str:
    """
    Vertex AI Search용 ServingConfig 리소스 경로를 생성합니다.

    Args:
        engine_id (str): 검색 엔진 ID (예: 'ws-001')
        project_id (str, optional): GCP 프로젝트 ID (기본: 환경변수 PROJECT_ID)
        location (str, optional): 리전 (기본: 환경변수 LOCATION 또는 'global')
        serving_name (str, optional): 서빙 구성 이름 (기본: 'default_config')

    Returns:
        str: ServingConfig 리소스 경로
    """
    project_id = project_id or os.environ["PROJECT_ID"]
    location   = location   or os.environ.get("LOCATION", "global")
    return (
        f"projects/{project_id}/locations/{location}"
        f"/collections/default_collection/engines/{engine_id}"
        f"/servingConfigs/{serving_name}"
    )

def retrieve_snippets(
    workspace_id: str,
    search_query: str,
    page_size: int = 5,
    serving_config: str | None = None,
    location: str = "global"
) -> List[Dict[str, Any]]:
    """
    Vertex AI Search에서 snippet 기반의 검색을 수행합니다.

    Args:
        workspace_id (str): 검색 대상 워크스페이스 ID
        search_query (str): 사용자 쿼리
        page_size (int, optional): 검색 결과 수 (기본값: 5)
        serving_config (str, optional): 사용할 서빙 구성 리소스 경로
        location (str, optional): 리전 (기본값: "global")

    Returns:
        List[Dict[str, Any]]: 각 문서의 스니펫, 메타데이터, 점수를 담은 결과 리스트
    """
    # 여기서는 PROJECT_ID, ENGINE_ID, LOCATION이 반드시 있어야 하므로 KeyError로 바로 알 수 있음

    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # SummarySpec을 제거하고 SnippetSpec만 사용하도록 수정
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True,
            # 스니펫을 최대 3개까지 반환하도록 설정
            max_snippet_count=3
        ),
        # summary_spec은 제거됨
    )

    # 필터 문자열 구성
    filter_string = f'workspace_id: ANY("{workspace_id}")'
    print(f"Applying filter: {filter_string}")

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=page_size,
        filter=filter_string,
        content_search_spec=content_search_spec, # 수정된 content_search_spec 적용
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)

    # 스니펫 파싱
    snippets_result_list = []
    for result in response.results:
        document = result.document
        doc_struct_data = document.struct_data if hasattr(document, "struct_data") else {}
        doc_derived_struct_data = document.derived_struct_data if hasattr(document, "derived_struct_data") else {}

        snippets_texts = []
        snippets = doc_derived_struct_data.get("snippets", [])
        for snippet in snippets:
            text = snippet.get("snippet", "")
            snippets_texts.append(text)

        file_name = doc_struct_data.get("file_name", doc_struct_data.get("uri", "N/A"))
        uri = file_name

        doc_info = {
            "file_path": doc_struct_data.get("file_path", "N/A"),
            "file_type": doc_struct_data.get("file_type", "N/A"),
            "parsing_type": doc_struct_data.get("parsing_type", "N/A"),
            "file_name": file_name,
            "content": snippets_texts,
            "workspace_id": doc_struct_data.get("workspace_id", "N/A"),
            "chunk_number":doc_struct_data.get("chunk_number", 0),
            "page_number": doc_struct_data.get("page_number", 0),
            "sheet_name": doc_struct_data.get("sheet_name", ""),
            "relevance_score": result.model_scores.get("relevance_score", 0.0).values[0]
                if hasattr(result, "model_scores") and result.model_scores and "relevance_score" in result.model_scores
                else 0.0
        }
        snippets_result_list.append(doc_info)

    return snippets_result_list


def retrieve_chunks(
    workspace_id: str,
    search_query: str,
    page_size: int = 5,
    serving_config: str | None = None,
    location: str = "global"
) -> List[Dict[str, Any]]:
    """
    Vertex AI Search에서 chunk 기반(semantic chunk) 검색을 수행합니다.

    Args:
        workspace_id (str): 검색 대상 워크스페이스 ID
        search_query (str): 사용자 쿼리
        page_size (int, optional): 검색 결과 수 (기본값: 5)
        serving_config (str, optional): 사용할 서빙 구성 리소스 경로
        location (str, optional): 리전 (기본값: "global")

    Returns:
        List[Dict[str, Any]]: 각 청크의 content와 관련 메타데이터를 담은 리스트
    """
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    client = discoveryengine.SearchServiceClient(client_options=client_options)

    filter_string = f'workspace_id: ANY("{workspace_id}")'

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=page_size,
        filter=filter_string,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            search_result_mode=(
                discoveryengine.SearchRequest.ContentSearchSpec
                .SearchResultMode.CHUNKS
            )
        ),
    )

    response = client.search(request=request)

    result_list: List[Dict[str, Any]] = []
    for result in response.results:
        chunk = result.chunk
        content = chunk.content or ""
        struct = chunk.document_metadata.struct_data if chunk.document_metadata and chunk.document_metadata.struct_data else {}

        result_list.append({
            "content": content,
            "file_type": struct.get("file_type", ""),
            "parsing_type": struct.get("parsing_type", ""),
            "file_name": struct.get("file_name", ""),
            "file_path": struct.get("file_path", ""),
            "workspace_id": struct.get("workspace_id", ""),
            "chunk_number": struct.get("chunk_number", 0),
            "page_number": struct.get("page_number", 0),
            "sheet_name": struct.get("sheet_name", ""),
            "score": chunk.relevance_score
        })

    return result_list

