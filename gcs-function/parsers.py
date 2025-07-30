# parsers.py
import re, struct, zlib, olefile
import io
from io import BytesIO
import pandas as pd
import openai
import os
import pdfplumber
import fitz  
import asyncio, base64

# ─ OpenAI 기본(동기) ----------------------------------------------------------------
openai.api_key = os.environ["OPENAI_API_KEY"]
sync_client = openai.Client()               # 기존 요약용

# ─ OpenAI Vision 비동기 -------------------------------------------------------------
try:
    async_client = openai.AsyncClient()     # Vision 이미지 OCR용
except AttributeError:
    raise RuntimeError("openai>=1.14.0 버전을 사용하세요 (AsyncClient 지원)")

def parse_excel_sheet_table(df: pd.DataFrame, sheet_name: str, file_name: str = "Unknown File") -> str:
    """
    DataFrame(표 시트)을 분석하여 칼럼 정보와 OpenAI LLM을 통한 데이터 전체 요약을 생성하고
    하나의 텍스트로 반환합니다.

    Args:
        df (pd.DataFrame): 분석할 DataFrame.
        sheet_name (str): 현재 처리 중인 시트 이름.
        file_name (str): 원본 Excel 파일의 이름 (요약에 포함될 수 있음).

    Returns:
        str: 칼럼 정보 (전체)와 시트 요약(LLM) 결과 텍스트
    """
    # 1) 칼럼 목록 생성
    columns_str = ", ".join(df.columns)
    
    # 2) df.info() 문자열화
    buf = io.StringIO()
    df.info(buf=buf, verbose=True, show_counts=True)
    info_str = buf.getvalue()

    # 3) OpenAI LLM을 통한 요약 생성
    prompt = f"""
        [파일] {file_name}
        [시트] {sheet_name}
        [DataFrame info]
        {info_str}

        위 정보를 바탕으로 이 시트를 간단히 요약해 주세요.
        """
    response = sync_client.chat.completions.create(
        model="gpt-4.1",  
        messages=[
            {"role": "system", "content": "아래 DataFrame 정보를 참고해서 시트를 요약해 주세요. 간단하게 한국어로 3~4줄로 써 주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    result_text = (
        "[칼럼 목록]\n"
        + columns_str
        + "\n\n[시트 요약]\n"
        + response.choices[0].message.content.strip()
    )
    return result_text

def parse_excel_sheet_text(df: pd.DataFrame) -> str:
    """
    비구조/텍스트 위주 엑셀 시트의 값을 모두 하나의 문자열로 합쳐 반환.

    Args:
        df (pd.DataFrame): 엑셀 데이터프레임.

    Returns:
        str: 결측치/공백/NaN 제거 후 하나로 합쳐진 텍스트.
    """
    flat_vals = df.values.flatten()
    cleaned = [
        str(v) for v in flat_vals
        if pd.notna(v) and (s := str(v).strip()) and s.lower() != "nan"
    ]
    # 연속 공백 제거
    text = re.sub(r"\s+", " ", " ".join(cleaned)).strip()
    return text


def extract_paragraphs(buf: bytes, offset=0):
    """
    HWP 바이너리 본문에서 문단별 텍스트 추출(재귀 파싱, 표는 <표> 태그 반환).

    Args:
        buf (bytes): HWP 파일의 바이너리 데이터.
        offset (int, optional): 파싱 시작 위치(기본: 0).

    Returns:
        list[str]: 문단별 텍스트 리스트.
    """
    parts, i, n = [], offset, len(buf)
    while i < n:
        hdr  = struct.unpack_from("<I", buf, i)[0]
        tag  = hdr & 0x3FF
        size = (hdr >> 20) & 0xFFF
        data = buf[i+4 : i+4+size]

        if tag == 67:  # 일반 문단
            parts.append(data.decode("utf-16le", errors="ignore"))
        elif tag == 61:  # 표 태그
            parts.append("<표>")
            parts.extend(extract_paragraphs(data)) # 표 내부 재귀 파싱
        elif tag in {17, 55, 71}:  # 각주, 헤더 등
            parts.extend(extract_paragraphs(data))
        i += 4 + size
    return parts

def parse_hwp_bytes(file_bytes: bytes) -> str:
    """
    HWP 파일 바이트 데이터를 받아 텍스트 추출 (OLE 바이너리 방식, .hwp v5)

    Args:
        file_bytes (bytes): GCS 등에서 받은 .hwp 원본 바이너리 데이터

    Returns:
        str: 전체 텍스트(문단·표 구조 포함)

    Raises:
        Exception: 파일 포맷 오류, 섹션 파싱 실패 등
    """
    # OLEFile은 파일객체 또는 BytesIO 지원
    with olefile.OleFileIO(BytesIO(file_bytes)) as f:
        header = f.openstream("FileHeader").read()
        zipped = (header[36] & 1) == 1  # 압축 여부

        # BodyText/SectionN 스트림 추출
        sections = []
        for p in f.listdir():
            if len(p) == 2 and p[0] == "BodyText":
                m = re.fullmatch(r"Section(\d+)", p[1])
                if m:
                    sections.append((int(m.group(1)), "/".join(p)))
        if not sections:
            raise RuntimeError("섹션 스트림을 찾지 못했습니다.")

        # 섹션 순서대로 병합
        paras = []
        for _, path in sorted(sections):
            raw = f.openstream(path).read()
            buf = zlib.decompress(raw, -15) if zipped else raw
            paras.extend(extract_paragraphs(buf))

        # 특수문자 필터링 및 텍스트 결합
        text = "\n".join(paras)
        text = re.sub(r"[^\x20-\x7E\uAC00-\uD7A3]", "", text).strip()
        return text


def parse_pdf_text(file_bytes: bytes) -> str:
    """
    pdfplumber를 사용해 PDF 텍스트를 추출합니다. 스캔 이미지 PDF에는 적합하지 않습니다.

    Args:
        file_bytes (bytes): PDF 파일의 원본 바이트 데이터.

    Returns:
        str: 추출된 전체 텍스트. 실패 시 빈 문자열.
    """

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
            page_texts = [text.strip() for text in page_texts]
        full_text = "\n\n".join(filter(None, page_texts))
        return full_text
    except Exception as e:
        print(f"[ERROR] pdfplumber 텍스트 추출 중 오류 발생: {e}", flush=True)
        return "" # 텍스트 추출 실패 시 빈 문자열 반환

async def _vision_ocr_page(img_bytes: bytes, detail="low", model="gpt-4.1-mini") -> str:
    """
    단일 이미지(PNG)를 OpenAI Vision API로 OCR 수행.

    Args:
        img_bytes (bytes): OCR 대상 이미지 (PNG).
        detail (str): detail 수준 ("low" 또는 "high").
        model (str): 사용할 Vision 모델.

    Returns:
        str: OCR 결과 텍스트 (마크다운 표 포함).
    """


    data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
    rsp = await async_client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {   # 1) ‘OCR 전용’ 역할
                "role": "system",
                "content": (
                    "You are an OCR engine. "
                    "Return every visible character exactly as it appears. "
                    "No explanations."
                )
            },
            {   # 2) 사용자 지시 + 이미지 (low 해상도)
                "role": "user",
                "content": [
                    {   # 2‑A) 텍스트 지시
                        "type": "text",
                        "text": (
                            "이미지 전체를 위→아래·왼→오 순서로 읽어줘. "
                            "표는 Markdown 파이프(|) 표로 재현하고, "
                            "다른 텍스트는 줄 바꿈을 유지해. "
                            "출력은 **단일 ```text 블록** 안에 넣고, 추가 설명 금지."
                        )
                    },
                    {   # 2‑B) 이미지, detail='low'
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": detail}
                    }
                ]
            }
        ]
    )
    return rsp.choices[0].message.content.strip()

async def _async_parse_pdf_ocr(file_bytes: bytes,
                                dpi=240,
                                detail="low",
                                model="gpt-4.1-mini",
                                max_conc=5) -> str:
    """
    PDF를 페이지 단위로 Vision OCR 수행. 비동기 방식으로 각 페이지 병렬 처리.

    Args:
        file_bytes (bytes): PDF 파일의 원본 바이트 데이터.
        dpi (int, optional): 렌더링 해상도. 기본값은 240.
        detail (str, optional): Vision detail 옵션. 기본값은 "low".
        model (str, optional): Vision OCR에 사용할 모델. 기본값은 "gpt-4.1-mini".
        max_conc (int, optional): 병렬 처리 최대 개수. 기본값은 5.

    Returns:
        str: 페이지별 OCR 결과를 합친 문자열.
    """

    doc   = fitz.open(stream=file_bytes, filetype="pdf")
    zoom  = dpi / 72
    sem   = asyncio.Semaphore(max_conc)
    texts = [None] * len(doc)

    async def worker(i):
        async with sem:
            pix   = doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_b = pix.tobytes("png")
            txt   = await _vision_ocr_page(img_b, detail, model)
            texts[i] = f"[Page {i+1}]\n{txt}"

    await asyncio.gather(*(worker(i) for i in range(len(doc))))
    doc.close()
    return "\n\n".join(texts)   

def parse_pdf_ocr(file_bytes: bytes,
                    dpi: int = 240,
                    detail: str = "low",
                    model: str = "gpt-4.1-mini",
                    max_conc: int = 2) -> str:
    """
    Vision OCR 비동기 함수를 동기 코드에서 호출하기 위한 래퍼 함수입니다.

    Args:
        file_bytes (bytes): PDF 파일의 원본 바이트 데이터.
        dpi (int, optional): 렌더링 해상도. 기본값은 240.
        detail (str, optional): OCR detail 수준. 기본값은 "low".
        model (str, optional): 사용할 Vision 모델. 기본값은 "gpt-4.1-mini".
        max_conc (int, optional): 최대 동시 OCR 처리 개수. 기본값은 2.

    Returns:
        str: 전체 페이지의 OCR 결과.
    """

    try:
        loop = asyncio.get_running_loop()
        # 노트북 / Cloud Functions Gen2 처럼 루프가 이미 있으면
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(
            _async_parse_pdf_ocr(file_bytes, dpi, detail, model, max_conc)
        )
    except RuntimeError:
        # 일반 스크립트 / FastAPI 백그라운드 등
        return asyncio.run(
            _async_parse_pdf_ocr(file_bytes, dpi, detail, model, max_conc)
        )


