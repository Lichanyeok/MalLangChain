from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Embedding
# Embedding 모델 리턴
# OpenAIEmbedding , Upstage 등
class Embedding():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    def getEmbedding(self):
        return self.embeddings


# LLM 객체
class LLMInstance():
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        return

    llm = any
    def getOpenAILLM(self):
        llm = ChatOpenAI(api_key = self.OPENAI_API_KEY)
        return llm
    def getLlamaLLM(self):
        pipe = pipeline(
            "text-generation",
            model=os.getenv("LLAMA_MODEL_NAME"),
            max_new_tokens=256,
            temperature=0.7,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
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


