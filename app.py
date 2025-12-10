import config
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx

class ToolCallingAgent:
    def __init__(self, llm):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = tool_map

    def run(self, query: str) -> str:
        # Step 1: Initial user message
        chat_history = [HumanMessage(content=query)]

        # Step 2: LLM chooses tool
        response = self.llm_with_tools.invoke(chat_history)
        if not response.tool_calls:
            return response.contet # Direct response, no tool needed
        # Step 3: Handle first tool call
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # Step 4: Call tool manually
        tool_result = self.tool_map[tool_name].invoke(tool_args)

        # Step 5: Send result back to LLM
        tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
        chat_history.extend([response, tool_message])

        # Step 6: Final LLM result
        final_response = self.llm_with_tools.invoke(chat_history)
        return final_response.content

openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key = config.OPENAI_API_KEY,
)


watsonx_llm = ChatWatsonx(
    model_id = "ibm/granite-3-3-8b-instruct",
    url = f"https://{config.WATSONX_REGION}.ml.cloud.ibm.com",
    project_id = config.WATSONX_PROJECT_ID,
    apikey = config.WATSONX_API_KEY,
)

@tool
def add(a: int, b: int) -> int:
    """
    Add a and b.
    
    Args:
        a (int): first integer to be added
        b (int): second integer to be added

    Return:
        int: sum of a and b
    """
    return a + b

@tool
def subtract(a: int, b:int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b."""
    return a * b

tool_map = {
    "add": add, 
    "subtract": subtract,
    "multiply": multiply
}

tools = [add, subtract, multiply]
my_agent = ToolCallingAgent(watsonx_llm)

print(my_agent.run("one plus 2"))

print(my_agent.run("one - 2"))

print(my_agent.run("three times two"))