from fastapi import APIRouter,Query
from app.api.file import EMBEDDING_LIST
from app.utils.chunk import text_spliter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
from pinecone import Pinecone
import os

router = APIRouter()
load_dotenv()

# LLM 객체
class LLMInstance():
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        return

    llm = any
    def getOpenAILLM(self):
        llm = ChatOpenAI(api_key = self.OPENAI_API_KEY)
        return llm

# Pinecone 객체
# 1. instance 리턴
# 2. index 리턴
class LangchainPinecone():
    def __init__(self):
        self.pik = os.getenv("PINECONE_API_KEY")
        self.pi = os.getenv("PINECONE_INDEX_NAME")
        self.pc = Pinecone(api_key=self.pik)

    def getPineconeInstane(self):
        return self.pc
    def getPineconeIndex(self):
        return self.pc.Index(self.pi)
    def getPineconeIndexName(self):
        return self.pi
    def getPineconeApikey(self):
        return self.pik

# Embedding
# Embedding 모델 리턴
# OpenAIEmbedding , Upstage 등
class Embedding():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    def getEmbedding(self):
        return self.embeddings

@router.get("/embedding")
async def file_embedding():
    for el in EMBEDDING_LIST:
        document = el["document"]
        chunks = text_spliter(document)

        lpi = LangchainPinecone()
        index = lpi.getPineconeIndex()

        embeddings = Embedding().getEmbedding()
        vectors = embeddings.embed_documents(chunks)

        to_upsert = [
            {
                "id":f"filename_{i}",
                "values":vector,
                "metadata":{"content":chunk}
            }
            for i , (chunk, vector) in enumerate(zip(chunks,vectors))
        ]

        index.upsert(to_upsert)

    return EMBEDDING_LIST

@router.get("/ffff")
def _retrieve_context(query:str):
    lpi = LangchainPinecone()
    embedding = Embedding()
    vs = PineconeVectorStore(
        index_name = lpi.getPineconeIndexName(),
        text_key="content",
        embedding=embedding.getEmbedding()
    )
    result = vs.similarity_search(query=query,k=4)
    return result

@router.get("/chat_prompt")
def chat_prompt(query:str = Query(..., description="사용자 질문 쿼리",max_length=10000)):
    print("Query : ", query)
    prompt = PromptTemplate(
        template = "너는 사용자의 질문에 답변하는 아주 훌륭한 도우미야 question이 들어온다면 context를 참조하여 답변을 해줘."
                   "- question : '{question}' "
                   "- context : '{context}'",
        input_variables=["question","context"]
    )

    li = LLMInstance()
    llm = li.getOpenAILLM()

    chain = LLMChain(llm=llm, prompt = prompt)
    result = chain.invoke({"question":query, "context":_retrieve_context(query)})
    return result["text"]


