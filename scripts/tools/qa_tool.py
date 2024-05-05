from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from scripts.tools.utils import pass_content_docs
from langchain.agents import Tool

def get_qa_tool(retriever, llm):
    qa_prompt = PromptTemplate(
        template="""
        Take a deep breath and answer to the following user question: 
        "{question}"
        
        Answer to this question, if you know the topic, otherwise ask for more details
        """,
        input_variables=["input", "question"],
    )

    qa_chain = (
        {'input': retriever | pass_content_docs, 'question': RunnablePassthrough()}
        | qa_prompt 
        | llm
    )

    qa_tool = Tool(name='Question Answering', 
                                func=qa_chain.invoke, 
                                description='Given a question abour wikipedia information, find and write the answer to the question')

    return qa_tool