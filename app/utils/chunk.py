from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile


async def _pdf_loader(read_file, filetype) :
    #UploadFile 자체를 로드할 수 없으므로, 임시파일 생성 -> 문서화 -> 임시파일 제거 순으로 처리
    with tempfile.NamedTemporaryFile(suffix=filetype,delete=False) as tmp:
        tmp.write(read_file)
        tmp_path = tmp.name # 임시파일 경로

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)

    txt_document = " ".join(doc.page_content.replace("\n", " ") for doc in documents)
    return txt_document

async def _docx_loader():
    return

async def load_document(read_file, filetype:str) -> []:
    document = []
    if ".pdf" == filetype:
        document = await _pdf_loader(read_file, filetype)

    return document


def text_spliter(document):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = spliter.split_text(document)
    return chunks