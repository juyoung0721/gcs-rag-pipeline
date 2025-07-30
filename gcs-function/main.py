# main.py
import os
from cloudevents.http import CloudEvent
import functions_framework
from processing import process_file
from indexing import indexing_chunks

# 환경변수 (필수)
PROJECT_ID   = os.environ["PROJECT_ID"]
DATASTORE_ID = os.environ["DATASTORE_ID"]  

@functions_framework.cloud_event
def gcs_cloudevent(event: CloudEvent):
    """
    GCS object.finalize → 파일 파싱 → 결과 로그
    CloudEvent 시그니처(인자 1개) 사용
    """
    data = event.get_data()
    bucket = data["bucket"]
    name   = data["name"]                      # 객체 전체 경로 (ws_12345/xxx.pdf 등)

    gcs_uri = f"gs://{bucket}/{name}"
    print(f"[INFO] gcs {name} - file upload !", flush=True)

    try:
        results = process_file(gcs_uri)

        if not results:
            print(f"[WARN] 파싱 결과가 비어 있습니다: {gcs_uri}", flush=True)
            return

        # (선택) 간단 프리뷰 로그
        first = results[0]
        preview = (first['content'] or "")[:200].replace("\n", " ")
        print(f"[DEBUG] first doc preview: {preview}", flush=True)

        # 2) 청크 생성 + Discovery Engine 업로드
        indexing_chunks(
            results,
            project_id=PROJECT_ID,
            location="global",
            datastore_id=DATASTORE_ID   # ENGINE_ID 사용 시 indexing.py에서 자동 선택
        )

    except Exception as e:
        # Error Reporting에서 잡히도록 예외 재발생
        print(f"[ERROR] {gcs_uri} 처리 중 예외: {e}", flush=True)
        raise
