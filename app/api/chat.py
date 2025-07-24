from fastapi import APIRouter,Query
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.api.instance import Embedding,LLMInstance,LangchainPinecone

router = APIRouter()

def _retrieve_context(query:str):
    lpi = LangchainPinecone()
    embedding = Embedding()
    vs = PineconeVectorStore(
        index_name = lpi.getPineconeIndexName(),
        text_key="content",
        embedding=embedding.getEmbedding()
    )
    result = vs.similarity_search(query=query,k=3)
    return result

@router.get("/chat_prompt")
def chat_prompt(query:str = Query(..., description="사용자 질문 쿼리",max_length=10000) ,
                model:str=Query(default="openai",description="llm 모델 설정")):

    prompt = PromptTemplate(
        template = "너는 사용자의 질문에 답변하는 아주 훌륭한 도우미야 question이 들어온다면 context를 참조하여 답변을 해줘."
                   "- question : '{question}' "
                   "- context : '{context}'",
        input_variables=["question","context"]
    )

    li = LLMInstance()
    llm = any
    if model == "openai":
        llm = li.getOpenAILLM()
    elif model == "llama":
        llm = li.getLlamaLLM()

    chain = LLMChain(llm=llm, prompt = prompt)
    result = chain.invoke({"question":query, "context":_retrieve_context(query)})
    return result["text"]


