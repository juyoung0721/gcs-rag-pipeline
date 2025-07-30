from retrieval import retrieve_chunks, retrieve_snippets
from excel_analysis import load_sheet, llm_df_to_code, execute_llm_code
import openai
import os

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.Client()
except openai.error.InvalidRequestError as e:
    print(f"OpenAI API 요청 오류: {e}")
except KeyError:
    print("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다. API 키를 직접 입력하거나 환경 변수를 설정해주세요.")



def final_answer_prompt(query, references):
    """
    references(근거 리스트)와 질의(query)를 받아
    LLM에 전달할 최종 자연어 프롬프트 문자열을 생성합니다.

    Args:
        query (str): 사용자 질문(질의)
        references (list[dict]): 각 근거 자료(ref) 딕셔너리 리스트
            - type: "excel_table" 또는 기타 parsing_type
            - file_name, sheet_name, content 등 포함

    Returns:
        str: LLM에 입력할 프롬프트 전체 텍스트
    """
    prompt = (
        f"아래는 사용자 질문과 참고자료 목록입니다.\n"
        f"## 질문: {query}\n"
        f"## 참고자료:\n"
    )
    for ref in references:
        if ref["parsing_type"] == "excel_table":
            prompt += f"- [엑셀] {ref['file_name']} - 시트:{ref['sheet_name']} 분석 결과: {ref['content']}\n"
        else:
            summary = ref.get('content', '')
            prompt += f"- [텍스트] {ref['file_name']}: {summary}...\n"
    prompt += "\n위 참고자료와 질문을 바탕으로 정확하고 근거 있는 답변을 생성하세요."

    
    return prompt

def llm_answer(
    mode: str,                 # "snippet" or "chunk"
    workspace_id: str,
    search_query: str,
    page_size: int = 5,
    model: str = "gpt-4.1",
    serving_config: str | None = None,
):
    """
    Vertex AI Search 기반으로 workspace/질문/모드에 따라 검색 및 references 생성 후,
    LLM(OpenAI 등)으로 최종 자연어 답변을 도출합니다.

    Args:
        mode (str): 검색 모드 ("snippet" 또는 "chunk")
        workspace_id (str): 검색 대상 워크스페이스 ID
        search_query (str): 사용자 질문(검색 쿼리)
        page_size (int, optional): 검색 결과 최대 개수 (기본: 5)
        model (str, optional): LLM 모델명 (기본: "gpt-4.1")

    Returns:
        str: LLM이 생성한 최종 답변 텍스트
    """
    # 1. 검색 실행
    if mode == "snippet":
        search_results = retrieve_snippets(
            workspace_id=workspace_id,
            search_query=search_query,
            page_size=page_size,
            serving_config=serving_config
        )
    else:
        search_results = retrieve_chunks(
            workspace_id=workspace_id,
            search_query=search_query,
            page_size=page_size,
            serving_config=serving_config
        )

    # 2. references 생성
    references = []
    print(f"chunks : {len(search_results)}개 검색 결과")
    for item in search_results:
        parsing_type = item.get("parsing_type", "")   # 사전에 저장되어 있다고 가정
        file_name = item.get("file_name")
        file_path = item.get("file_path", "")


        if parsing_type == "excel_table":
            # Excel 분석 경로
            sheet_name = item.get("sheet_name")
            df = load_sheet(file_path, sheet_name)
            code = llm_df_to_code(search_query, df)
            analysis_result = execute_llm_code(code, df)
            references.append({
                "parsing_type": "excel_table",
                "file_name": file_name,
                "sheet_name": sheet_name,
                "content": analysis_result
            })
        else:
            # 일반 텍스트/청크/snippet 경로
            content = item.get("content") or (item.get("snippets") and " ".join(item["snippets"]))
            references.append({
                "parsing_type": parsing_type,
                "file_name": file_name,
                "content": content
            })


    # 3. 최종 LLM Answer
    prompt = final_answer_prompt(search_query, references)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "아래 질문과 참고자료를 바탕으로 정확하고 간결한 답변을 작성하세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )


    return response.choices[0].message.content, references
