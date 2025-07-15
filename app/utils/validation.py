from tiktoken import encoding_for_model
import app.constants as Constants
import os

# 학습 가능한 파일인지 확장자 검사.
def check_file_type(file_type : str) -> {}:
    if not file_type in (".pdf",".txt",".doc",".docx"):
        return {
            "result" : False ,
            "result_message" : "적용 가능한 파일이 아닙니다."
        }
    return {
        "result": True,
        "result_message": "적용 가능한 파일입니다.",
    }

# 토큰 개수 검사.
def check_file_token(documents)->{}:
    encoder = encoding_for_model(Constants.DEFAULT_MODEL)
    txt_document = " ".join(doc.page_content.replace("\n", " ") for doc in documents)
    token_count = len(encoder.encode(txt_document))

    if Constants.MAX_TOKEN < token_count:
        result = {
            "result" : False ,
            "result_message" : f"토큰 개수 초과 : {token_count - Constants.MAX_TOKEN}"
        }
        return result

    result = {
        "result" : True ,
        "result_message" : "토큰 개수 통과."
    }
    return result