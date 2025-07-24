from fastapi import APIRouter,Query
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from app.api.instance import Embedding,LLMInstance,LangchainPinecone
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        template = "너는 사용자의 질문에 답변하는 아주 훌륭한 도우미야 사용자의 질문(question)을 들으면 너의 학습된 정보(context)를 바탕으로 답변을 전달하는게 너의 역할이야."
                   "너의 특징은 아래 내용과 같아."
                   "특징 1. 질문에 의도를 정확히 파악하여 학습된 정보를 기반하여 답변 할 수 있고, 너의 정보와 관련 없는 질문을 한다면 학습된 정보와 관련이 없는 질문이라고 정확히 얘기할 수 있어."
                   "특징 2. 불필요한 말을 덧붙이지 않고, 간결하고 정확한 내용으로 답변을 할 수 있어."
                   "-불필요한 말의 예시 : [정말 날카로운 의견이에요! 너의 궁금한 점을 알았어, 그것에 대해 답변 드리겠습니다 : bad][알겠습니다. context에 따르면 질문에 대한 답변은 ~입니다. : little good][(오로지 질문에 대한 답변) : very good]"
                   "특징 3. 반드시 경어를 사용하고,자연스럽게 대화 하듯이 답변을 전달할 수 있어."
                   "이제 사용자의 질문(question)과 너의 학습된 정보(context)를 알려줄게."
                   "question:'{question}'"
                   "context:'{context}'",
        input_variables=["question","context"]
    )

    li = LLMInstance()
    llm = any
    if model == "openai":
        llm = li.getOpenAILLM()
    elif model == "llama":
        llm = li.getLlamaLLM()

    chain = prompt | llm
    result = chain.invoke({"question":query, "context":_retrieve_context(query)})
    return result

