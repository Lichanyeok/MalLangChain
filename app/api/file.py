from fastapi import APIRouter, UploadFile,File
from typing import Dict , List
from app.utils.validation import check_file_type, check_file_token
from app.utils.chunk import load_document
import os


router = APIRouter()
FILE_CACHE : List[Dict[str,any]] = [] # 업로드 파일 리스트(파일명, 검증 결과)
EMBEDDING_LIST : List[Dict[str,any]] = [] # 실제 임베딩할 파일

@router.post("/upload_staging")
async def file_staging(files : list[UploadFile] = File(...)):
    for file in files:
        file_name = file.filename
        file_type = os.path.splitext(file.filename)[1].lower()
        read_file = await file.read()
        # 업로드 파일 확장자 검증
        file_check_result = check_file_type(file_type)
        if not file_check_result["result"]:
            FILE_CACHE.append({
                "filename":file_name ,
                "result_message" : file_check_result["result_message"],
            })
            continue

        # 입력받은 파일 문서화 처리
        document = await load_document(read_file,file_type)

        # 업로드 파일 토큰 초과 여부 검증
        file_check_result = check_file_token(document)
        if not file_check_result["result"]:
            FILE_CACHE.append({
                "filename":file.filename ,
                "result_message" : file_check_result["result_message"]
            })
            continue

        # 검증 통과 대상만 임베딩 대상 리스트업
        FILE_CACHE.append({
            "filename":file.filename ,
            "result_message" : "적용 가능한 파일입니다."
        })

        #0 : 처리대기, 1 : 처리완료 , 2 : 처리실패
        EMBEDDING_LIST.append({
            "filename":file_name ,
            "document": document ,
            "status" : 0 ,
            "message" : "처리 대기중입니다."
        })

    return EMBEDDING_LIST