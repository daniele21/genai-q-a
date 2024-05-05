from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import Tool

def get_general_tool(llm):
    general_prompt = PromptTemplate(
        template="""
        Remind the user that you are able to answer to the following tools: {tools}

        Answer to this general question of the user: "{question}" 
        
        """,
        input_variables=["tools", "question"],
    )

    general_chain = (
        {'tools': RunnablePassthrough(), 'question': RunnablePassthrough()}
        | general_prompt 
        | llm
    )

    general_tool = Tool(name='General Question', 
                        func=general_chain.invoke, 
                        description='This tool is to use when the user query is not associable with the other tools')

    return general_tool