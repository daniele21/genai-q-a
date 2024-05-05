from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from scripts.tools.utils import pass_content_docs
from langchain.agents import Tool

def get_code_tool(retriever, llm):
    code_prompt = PromptTemplate(
        template="""
        Take a deep breath and answer to the following user question, using python: 
        "{question}"
        
        Answer to this question, if you can, otherwise ask for more details
        """,
        input_variables=["input", "question"],
    )

    code_chain = (
        {'input': retriever | pass_content_docs, 'question': RunnablePassthrough()}
        | code_prompt 
        | llm
    )

    code_tool = Tool(name='Python Code Tool', 
                                func=code_chain.invoke, 
                                description='Tool use when the user ask for write python code and/or executing it')

    return code_tool