# processing.py
import pandas as pd
from google.cloud import storage
import os, re
from io import BytesIO
from parsers import parse_excel_sheet_table, parse_excel_sheet_text, parse_hwp_bytes, parse_pdf_text, parse_pdf_ocr
import pandas as pd
from typing import List, Dict
import fitz

def is_table_sheet(df: pd.DataFrame,
                    min_cols: int = 2,
                    min_rows: int = 2,
                    min_fill_ratio: float = 0.1) -> bool:
    """
    주어진 DataFrame이 '표 형식 시트'인지 여부를 판별합니다.

    - 최소 열/행 수
    - 셀 채움 비율
    - 모든 열 이름이 'Unnamed'인 경우는 제외

    Args:
        df (pd.DataFrame): 판별할 엑셀 시트 데이터프레임
        min_cols (int, optional): 최소 열 개수 기준. 기본값은 2.
        min_rows (int, optional): 최소 행 개수 기준. 기본값은 2.
        min_fill_ratio (float, optional): 채워진 셀 비율 최소 기준 (0~1). 기본값은 0.1.

    Returns:
        bool: 표 형식 시트 여부
    """

    # 1) 구조 최소 요건
    if df.shape[1] < min_cols or df.shape[0] < min_rows:
        return False

    # 2) 헤더명이 전부 Unnamed 면 표 아님
    if all(str(c).startswith("Unnamed") for c in df.columns):
        return False

    # 3) 값이 들어있는 셀 비율(≥ 10 %) → 주석 시트는 거의 빈칸
    total_cells = df.shape[0] * df.shape[1]
    filled_cells = df.notna().sum().sum()
    if filled_cells / total_cells < min_fill_ratio:
        return False

    return True

def is_scanned_pdf(file_bytes: bytes, 
                    min_char_count: int = 10,
                    check_pages: int = 3) -> bool:
    """
    PDF가 스캔 이미지 기반인지 여부를 판별합니다.

    Args:
        file_bytes (bytes): PDF 파일의 바이트 데이터
        min_char_count (int, optional): 텍스트 PDF로 판단할 최소 문자 수. 기본값은 10.
        check_pages (int, optional): 확인할 페이지 수. 기본값은 3.

    Returns:
        bool: True면 스캔 PDF, False면 텍스트 기반 PDF
    """

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    check_pages = min(check_pages, len(doc))
    for i in range(check_pages):
        page = doc.load_page(i)
        text = page.get_text("text")
        if len(text.strip()) >= min_char_count:
            doc.close()
            return False  # 텍스트 PDF
    doc.close()
    return True  # 스캔 PDF

def detect_file_type(file_path: str) -> str:
    """
    파일 경로나 이름에서 확장자를 추출하여 파일 유형을 반환합니다.

    Args:
        file_path (str): 파일 경로나 이름

    Returns:
        str: 'pdf', 'hwp', 'xlsx' 중 하나

    Raises:
        ValueError: 지원하지 않는 확장자일 경우
    """

    ext = os.path.splitext(file_path)[-1].lower().replace('.', '')
    if ext in {"pdf", "hwp", "xlsx"}:
        return ext
    raise ValueError(f"지원하지 않는 파일 확장자: {ext}")

def extract_workspace_id(blob_name: str) -> str | None:
    """
    GCS 경로에서 workspace_id(ws_12345 형식)를 추출합니다.

    Args:
        blob_name (str): GCS의 객체 이름 (예: 'ws_12345/abc.xlsx')

    Returns:
        str | None: workspace_id 문자열 또는 찾지 못한 경우 None
    """

    m = re.match(r"(ws_\d+)/", blob_name)
    return m.group(1) if m else None

def process_file(gcs_uri: str) -> List[Dict]:
    """
    GCS URI에서 파일을 다운로드하여 파일 형식에 따라 자동으로 파싱합니다.

    - Excel(xlsx): 각 시트를 표 여부로 구분하여 요약 또는 텍스트 추출
    - HWP: 전체 텍스트 파싱
    - PDF: 텍스트 PDF vs 스캔 PDF를 구분 후 적절한 파서 사용

    Args:
        gcs_uri (str): 'gs://bucket_name/path/to/file' 형식의 GCS URI

    Returns:
        List[Dict]: 각 문서 또는 시트별로 파싱된 텍스트와 메타데이터를 포함하는 딕셔너리 리스트

    Raises:
        ValueError: URI 형식 오류 또는 확장자 미지원 시
    """


    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri는 'gs://'로 시작해야 합니다.")
    
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    file_bytes = blob.download_as_bytes()
    file_name = os.path.basename(blob_name)
    file_type = detect_file_type(file_name)
    workspace_id = extract_workspace_id(blob_name)

    results: List[Dict] = []

    if file_type == "xlsx":
        print(f"file process start : {file_name}")
        xl = pd.ExcelFile(BytesIO(file_bytes))

        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            # 여기에 is_table_sheet 사용해서 판별 가능
            if is_table_sheet(df):
                text = parse_excel_sheet_table(df, sheet_name=sheet, file_name=file_name)
                parsing_type = "excel_table"
            else:
                text = parse_excel_sheet_text(df)
                parsing_type = "excel_text"
            
            results.append({
                "content": text,
                "metadata": {
                    "file_path": gcs_uri,
                    "file_name": file_name,
                    "file_type": "xlsx",
                    "parsing_type": parsing_type,
                    "sheet_name": sheet,
                    "workspace_id": workspace_id
                }
            })

    # HWP 파싱
    elif file_type == "hwp":
        full_text = parse_hwp_bytes(file_bytes)
        results.append({
            "content": full_text,
            "metadata": {
                "file_path": gcs_uri,
                "file_name": file_name,
                "file_type": "hwp",
                "parsing_type": "hwp",
                "workspace_id": workspace_id
            }
        })
    
    elif file_type == "pdf":
        if is_scanned_pdf(file_bytes):
            full_text = parse_pdf_ocr(file_bytes)
            results.append({
                "content": full_text,
                "metadata": {
                    "file_path": gcs_uri,
                    "file_name": file_name,
                    "file_type": "pdf",
                    "parsing_type": "pdf_ocr",
                    "workspace_id": workspace_id
                }
            })

        else:
            full_text = parse_pdf_text(file_bytes)
            results.append({
                "content": full_text,
                "metadata": {
                    "file_path": gcs_uri,
                    "file_name": file_name,
                    "file_type": "pdf",
                    "parsing_type": "pdf_text",
                    "workspace_id": workspace_id
                }
            })
    
    return results