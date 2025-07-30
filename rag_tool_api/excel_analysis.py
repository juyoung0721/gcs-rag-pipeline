# excel_analysis.py
import pandas as pd
from google.cloud import storage
from io import BytesIO
import json
import openai
import os
import subprocess
import textwrap
import sys

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.Client()
except openai.error.InvalidRequestError as e:
    print(f"OpenAI API 요청 오류: {e}")
except KeyError:
    print("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다. API 키를 직접 입력하거나 환경 변수를 설정해주세요.")


def load_sheet(gcs_uri: str, sheet_name: str) -> pd.DataFrame:
    """
    GCS URI에서 Excel 파일을 다운로드하고, 지정한 시트를 DataFrame으로 반환합니다.

    Args:
        gcs_uri (str): 'gs://bucket_name/path/to/file.xlsx' 형식의 GCS URI
        sheet_name (str): 불러올 Excel 시트 이름

    Returns:
        pd.DataFrame: 지정한 시트의 데이터프레임

    Raises:
        ValueError: GCS URI가 'gs://'로 시작하지 않을 경우
    """

    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri는 'gs://'로 시작해야 합니다.")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()
    return pd.read_excel(BytesIO(data), sheet_name=sheet_name)

def df_metadata(df: pd.DataFrame, head_rows=5) -> dict:
    """
    DataFrame의 구조, 칼럼별 요약 통계, 샘플 데이터(헤드)를 반환합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        head_rows (int, optional): 샘플로 추출할 앞쪽 행 개수 (기본값: 5)

    Returns:
        dict: 
            - shape: (행, 열) 튜플
            - schema: 칼럼별 타입, 결측치, 통계 등 리스트
            - sample_rows: 헤드 행을 dict 리스트로 반환

    Example:
        meta = df_metadata(df)
        print(meta["shape"], meta["schema"][0], meta["sample_rows"][0])
    """
    meta = {
        "shape": df.shape,  # (rows, cols)
        "schema": [],
        "sample_rows": df.head(head_rows).to_dict(orient="records"),
    }
    for col in df.columns:
        info = {
            "name": col,                      # 칼럼명
            "dtype": str(df[col].dtype),      # 데이터타입(str)
            "non_null": int(df[col].notna().sum())  # 결측값이 아닌 데이터 개수
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            # 숫자형이면 기초 통계 제공
            info.update(
                min=df[col].min(),
                max=df[col].max(),
                mean=float(df[col].mean())
            )
        else:
            # 범주형 등은 유니크 값 샘플링
            uniq = df[col].dropna().unique()
            info.update(
                n_unique=len(uniq),
                sample_values=uniq[:5].tolist()
            )
        meta["schema"].append(info)
    return meta


def llm_df_to_code(question: str, df: pd.DataFrame, model = "gpt-4.1") -> str:
    """
    DataFrame의 메타데이터와 사용자 질문을 바탕으로,
    LLM(OpenAI)을 이용해 pandas 코드(코드만, 설명 없이)를 생성합니다.

    규칙:
    - 필요한 전처리/집계 포함
    - 결과를 result 변수에 저장
    - 마지막 줄은 print(result) (f-string, 질문 맥락의 한글 레이블)
    - 코드 외 설명/주석/불필요한 출력 없음

    Args:
        question (str): 사용자 질문 (예: '부서별 평균 매출을 구해줘')
        df (pd.DataFrame): 분석할 데이터프레임

    Returns:
        str: LLM이 생성한 pandas 코드(코드블록 기호·주석 없이)
    """
    # 1) DataFrame 메타데이터를 JSON 문자열로 변환
    meta_json = json.dumps(df_metadata(df), ensure_ascii=False, default=str)

    # 2) LLM 프롬프트 준비
    system_prompt = """
        너는 pandas 전문가야. 아래 JSON은 DataFrame(df)의 메타데이터다.
        - 사용자 질문을 해결하는 pandas 코드를 작성하라.
        규칙:
        1. 필요한 전처리·집계 코드를 작성
        2. 최종 결과를 result 변수에 저장
        3. 마지막 줄엔 print(result)
        4. 코드 외 설명·주석 금지
        5. print는 f-string을 사용해, **result 값마다 질문 맥락에 맞는 한국어 레이블과 함께**
        ‘레이블 : 값’ 형식으로 간결하게 출력한다.
        """
    user_prompt = f"## DataFrameMeta ##\n```json\n{meta_json}\n```\n\n## Question ##\n{question}"
    
    response = client.responses.create(
        model=model,
        instructions= system_prompt,
        input=user_prompt
    )
    code_block = response.output_text

    return code_block.replace("```python", "").replace("```", "").strip()


def execute_llm_code(code_str: str, df: pd.DataFrame, timeout_sec: int = 5):
    """
    LLM이 생성한 pandas 코드를 서브프로세스에서 안전하게 실행합니다.

    - DataFrame은 JSON 형태로 stdin을 통해 전달
    - 코드 실행 후 print 결과를 stdout에서 수집
    - 코드 오류 또는 시간 초과 등을 처리하여 None 반환

    Args:
        code_str (str): 실행할 pandas 코드 문자열
        df (pd.DataFrame): 코드에서 사용할 데이터프레임
        timeout_sec (int, optional): 최대 실행 시간(초). 기본값은 5초

    Returns:
        str or None: 코드 실행 결과 (표준 출력), 오류 시 None

    Raises:
        subprocess.TimeoutExpired: 실행 시간 초과
        subprocess.CalledProcessError: 코드 실행 실패
        Exception: 예기치 못한 에러
    """

    df_json_input = df.to_json(orient='records').encode('utf-8')

    script_to_run = textwrap.dedent(f"""
import pandas as pd
import json
import sys
from io import StringIO

# stdin에서 JSON 데이터를 읽어 데이터프레임으로 변환
df_raw = sys.stdin.read()
df = pd.read_json(StringIO(df_raw), orient='records')

# LLM이 생성한 코드 (여기에 삽입됨)
{code_str}

    """)

    try:
        process = subprocess.run(
            [sys.executable, "-c", script_to_run],
            input=df_json_input,
            capture_output=True,
            check=True,
            timeout=timeout_sec,
        )
        # print문 결과(표준 출력)만 텍스트로 반환
        if process.stdout:
            return process.stdout.decode('utf-8').strip()
        else:
            print(f"서브프로세스가 출력을 반환하지 않았습니다. Stderr: {process.stderr.decode('utf-8')}")
            return None

    except subprocess.TimeoutExpired:
        print(f"오류: 코드 실행 시간 초과 ({timeout_sec}초).")
        return None
    except subprocess.CalledProcessError as e:
        print(f"오류: 서브프로세스 실행 실패 (종료 코드: {e.returncode}). Stderr: {e.stderr.decode('utf-8')}")
        return None
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        return None
