from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
import os
import re


def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


add_tool = Tool(name="add", description="Add a and b.", func=add)
multiply_tool = Tool(name="multiply", description="Multiply a and b.", func=multiply)
divide_tool = Tool(name="divide", description="Divide a by b.", func=divide)
tools = [add_tool, multiply_tool, divide_tool]
tools_by_name = {t.name: t for t in tools}

def get_model_with_tools():
    m = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return m.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]
    llm_calls: int


def llm_call(state: dict):
    return {
        "messages": [
            get_model_with_tools().invoke(
                [
                    SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."),
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: dict):
    result = []
    last = state["messages"][-1]
    for tool_call in getattr(last, "tool_calls", []) or []:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", []):
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()


def run_calculator_agent(question: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return _simple_calc(question)
    messages = [HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})
    out = result["messages"][-1]
    return getattr(out, "content", "")


def _simple_calc(q: str) -> str:
    s = q.strip().lower()
    m = re.search(r"add\s+(\d+)\s+(and|\+)\s+(\d+)", s)
    if m:
        return str(int(m.group(1)) + int(m.group(3)))
    m = re.search(r"multiply\s+(\d+)\s+(and|x|\*)\s+(\d+)", s)
    if m:
        return str(int(m.group(1)) * int(m.group(3)))
    m = re.search(r"divide\s+(\d+)\s+(by|/)\s+(\d+)", s)
    if m:
        b = int(m.group(3)) or 1
        return str(int(m.group(1)) / b)
    return "Unable to compute"
