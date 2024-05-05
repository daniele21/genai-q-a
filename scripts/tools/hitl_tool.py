from langchain.agents import load_tools


def get_human_tool(llm):
    human_in_the_loop_tool = load_tools(
        ["human"],
        llm=llm,
    )[0]

    return human_in_the_loop_tool