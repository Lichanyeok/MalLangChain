from fastapi import APIRouter
from app.api.file import EMBEDDING_LIST
from app.utils.chunk import text_spliter
from langchain.vectorstores import Pinecone as LangchaninPincone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import os

router = APIRouter()
load_dotenv()

@router.get("/listtest")
async def file_embedding():
    for el in EMBEDDING_LIST:
        document = el["document"]
        chunks = text_spliter(document)
            

        pik = os.getenv("PINECONE_API_KEY")
        pr = os.getenv("PINECONE_ENVIRONMENT")
        pi = os.getenv("PINECONE_INDEX_NAME")
        aik = os.getenv("OPENAI_API_KEY")

        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key =pik)
        index = pc.Index(pi)

        LangchaninPincone(
            index=index,
            documents=chunks,
            embedding=embeddings,
            index_name=pi
        )

    return EMBEDDING_LIST