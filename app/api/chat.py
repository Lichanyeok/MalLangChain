from fastapi import APIRouter,Query
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
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

    #reformulation
    reformulation_prompt = PromptTemplate(
        template="다음은 사용자의 질문입니다. : '{query}'"
                 "아래 자료를 기반하여 더 정확한 질문으로 만들어주세요."
                 "참고자료 : '{context}'",
        input_variables=["query","context"]
    )

    #main 프롬프트
    prompt = PromptTemplate(
        template = "당신은 사용자의 질문에 답변하는 아주 훌륭한 도우미입니다. 사용자의 질문(question)을 받고 학습된 정보(context)를 기반으로 한 답변을 전달하는게 당신의 역할입니다."
                   "당신의 특징은 아래와 같습니다."
                   "특징 1. 질문에 의도를 정확히 파악하여 학습된 정보를 기반하여 답변 할 수 있고, 학습된 정보에서 너무 많이 벗어난 질문에는 [해당 질문에는 답변할 수 없습니다.] 라고 얘기할 수 있습니다."
                   "특징 2. 불필요한 감탄사를 덧붙이지 않고 간결하고 정확한 내용으로 답변을 할 수 있습니다."
                   "특징 3. 반드시 경어를 사용하고, 자연스럽게 대화 하듯이 답변을 전달할 수 있습니다."
                   "이제 사용자의 질문(question)과 당신의 학습된 정보(context)를 알려드리겠습니다. 당신의 특징을 잘 살려 답변 해 주시기를 부탁합니다."
                   "question:'{question}'"
                   "context:'{context}'",
        input_variables=["question","context"]
    )

    #llm 세팅
    li = LLMInstance()
    llm = any
    if model == "openai":
        llm = li.getOpenAILLM()
    elif model == "llama":
        llm = li.getLlamaLLM()

    #체인구성 - S

    #reformulation chain
    reformulation_chain = reformulation_prompt | llm
    reformulation_result = reformulation_chain.invoke({"query":query, "context":_retrieve_context(query)})
    reformulation_query = reformulation_result.content

    # main chain
    chain = prompt | llm
    result = chain.invoke({"question":reformulation_query, "context":_retrieve_context(query)})
    #체인구성 - E
    print(reformulation_query)
    """
        q: 임금체불
        a: 해당 질문에는 답변할 수 없습니다.
        
        
        q : 임금체불
        rq:'임금체불'에 대한 법률적 규정과 관련된 질문을 해주세요.
        a : "해당 질문에 대한 법률적 규정은 근로기준법 제43조의2에 해당합니다. 
        이에 따르면 고용노동부장관은 임금을 지급하지 않는 사업주인 체불사업주를 공개 명단에 올릴 수 있습니다. 
        체불사업주는 지급하지 않은 임금, 보상금, 수당 등을 말하며, 3천만원 이상을 체불한 경우 공개 명단에 올릴 수 있습니다. 
        해당 사업주에게는 소명 기회가 주어지며, 공개 명단에 올라가면 법률에 따라 일정한 제재가 가해질 수 있습니다."
    """
    return result

