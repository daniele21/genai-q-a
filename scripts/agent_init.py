from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy import O
from tenacity import RetryError
from scripts.llm_load import load_llms
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

from scripts.tools.code_tool import get_code_tool
from scripts.tools.general_tool import get_general_tool
from scripts.tools.hitl_tool import get_human_tool
from scripts.tools.qa_tool import get_qa_tool
from langchain.agents import initialize_agent, AgentType


def init_agent(llm_path, split_params, emb_path, memory_params):
    agent_llm, qa_llm, code_llm, general_llm, summarizer_llm = load_llms(llm_path)

    # load dataset
    dataset = load_dataset("rag-datasets/mini_wikipedia", "text-corpus", split='passages')
    data = [x['passage'] for x in dataset]

    # Data Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_params['chunk_size'],
        chunk_overlap=split_params['chunk_overlap'],
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents(data)

    # Retriever
    embeddings = HuggingFaceEmbeddings(model_name=emb_path)
    faiss_db = FAISS.from_documents(documents=texts, embedding=embeddings)
    retriever = faiss_db.as_retriever()

    # Tools
    qa_tool = get_qa_tool(retriever, qa_llm)
    code_tool = get_code_tool(retriever, code_llm)
    human_tool = get_human_tool(agent_llm)
    general_tool = get_general_tool(llm=general_llm)

    tools = [qa_tool, 
            code_tool, 
            human_tool, 
            general_tool
            ]
    tools_format = {tool.name:tool.description for tool in tools}
    tool_names = [tool.name for tool in tools]
    
    # Agent
    memory = ConversationSummaryBufferMemory(llm=summarizer_llm, 
                                         max_token_limit=memory_params['max_token_limit'])

    conv_chain = ConversationChain(
        llm=agent_llm,
        memory=memory,
        verbose=True,
    )


    conv_agent = initialize_agent(
        tools,
        llm=conv_chain,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    return conv_agent
