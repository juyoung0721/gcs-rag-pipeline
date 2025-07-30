# indexing.py
import uuid
from typing import Any, Dict, List

from google.cloud import discoveryengine_v1beta as discoveryengine
from google.cloud.discoveryengine_v1beta.types import Document
from google.protobuf import struct_pb2

# ---------------------------------------------------------------------------
# 1. Chunk utilities
# ---------------------------------------------------------------------------


def chunk_dict_fixed(
    doc: Dict[str, Any],
    *,
    chunk_size: int = 512,
    overlap: int = 51,
) -> List[Dict[str, Any]]:
    """
    주어진 텍스트 문서를 고정 길이로 청크 단위로 분할합니다.

    - content를 chunk_size만큼 자르고, 각 청크 간 overlap만큼 겹쳐서 구성합니다.
    - 각 청크는 metadata와 함께 chunk_number 필드를 추가하여 반환됩니다.

    Args:
        doc (Dict[str, Any]): 'content'(str)와 'metadata'(dict)를 포함한 문서.
        chunk_size (int, optional): 각 청크의 최대 길이. 기본값은 512.
        overlap (int, optional): 청크 간 중첩되는 문자 수. 기본값은 51.

    Returns:
        List[Dict[str, Any]]: 각 청크를 담은 딕셔너리 리스트.
    
    Raises:
        KeyError: 'content' 또는 'metadata' 키가 doc에 없을 경우.
    """
    if "content" not in doc or "metadata" not in doc:
        raise KeyError("doc must have 'content' and 'metadata' keys")

    content: str = doc["content"] or ""
    metadata: Dict[str, Any] = doc["metadata"].copy()

    chunks: List[Dict[str, Any]] = []
    start: int = 0
    chunk_number: int = 1

    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]

        # shallow copy metadata and inject chunk_number
        chunk_meta = {**metadata, "chunk_number": chunk_number}
        chunks.append({"content": chunk_text, "metadata": chunk_meta})

        chunk_number += 1
        start += chunk_size - overlap

    return chunks

# ---------------------------------------------------------------------------
# 2. Vertex AI Search / Discovery Engine uploader
# ---------------------------------------------------------------------------

def upload_chunks_to_vertex_search(
    content, metadata,
    project_id, location, datastore_id
):
    """
    단일 청크를 Vertex AI Search (Discovery Engine)에 업로드합니다.

    Args:
        content (str): 업로드할 텍스트 내용.
        metadata (Dict[str, Any]): 문서에 포함할 메타데이터.
        project_id (str): GCP 프로젝트 ID.
        location (str): 리전 이름 (예: "us-central1").
        datastore_id (str): 데이터스토어 ID.

    Returns:
        bool: 업로드 성공 여부.

    Raises:
        Exception: 업로드 도중 오류 발생 시 내부 처리.
    """

    client = discoveryengine.DocumentServiceClient()
    parent = client.branch_path(project_id, location, datastore_id, "default_branch")

    try:
        content_bytes = content.encode('utf-8')

        # Document.Content 객체를 딕셔너리로 초기화
        # raw_bytes 필드와 mime_type 필드를 사용합니다.
        document_content_object = Document.Content(
            raw_bytes=content_bytes,
            mime_type="text/plain"
        )
        document_id = str(uuid.uuid4())

        struct_data_pb = struct_pb2.Struct()
        struct_data_pb.update(metadata)

        document_fields = {
            "id": document_id,
            "content": document_content_object,
            "struct_data": struct_data_pb,
        }
        document = Document(document_fields) 

        request = discoveryengine.CreateDocumentRequest(
            parent=parent,
            document=document,
            document_id=document_id,
        )

        print(f"데이터 스토어에 문서 '{document_id}' 추가 중...")
        response = client.create_document(request=request)
        print(f"문서 '{response.id}'가 성공적으로 추가/업데이트되었습니다.")
        return True

    except Exception as e:
        print(f"데이터 스토어에 문서 추가 중 오류 발생: {e}")
        return False
    
def indexing_chunks(json_data, project_id, location, datastore_id):
    """
    JSON 데이터를 고정 크기 청크로 나눈 후 각 청크를 Discovery Engine에 업로드합니다.

    Args:
        json_data (List[Dict[str, Any]]): 'content'와 'metadata'를 포함한 문서 리스트.
        project_id (str): GCP 프로젝트 ID.
        location (str): 리전 이름 (예: "us-central1").
        datastore_id (str): 업로드 대상 데이터스토어 ID.

    Returns:
        None
    """

    chunks = []
    for data in json_data:
        fixed_chunks = chunk_dict_fixed(data)
        chunks.extend(fixed_chunks)

    for chunk in chunks:
        content = chunk["content"]
        metadata = chunk["metadata"]

        success = upload_chunks_to_vertex_search(
            content, metadata,
            project_id=project_id,
            location=location,
            datastore_id=datastore_id
        )
        
        if success:
            print(f"✅ 청크 업로드 성공: {metadata['file_name']} (청크 번호: {metadata['chunk_number']})")
        else:
            print(f"❌ 청크 업로드 실패: {metadata['file_name']} (청크 번호: {metadata['chunk_number']})")