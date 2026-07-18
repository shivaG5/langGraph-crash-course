from typing import TypedDict, Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Global variable to store the document content
doc_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content.

    Args:
        content: The full updated document content.
    """
    global doc_content
    doc_content = content
    return f"Document has been updated successfully! The current content is:\n{doc_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """
    global doc_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(doc_content)
        print(f"\nDocument has been saved to: {filename}")
        return f"Document has been successfully saved to '{filename}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]
model = ChatOllama(model="qwen2.5:7b", temperature=0).bind_tools(tools)


def doc_agent(state: AgentState) -> AgentState:
    sys_prompt = SystemMessage(content=f"""You are Drafter, a document editing assistant.

Current document:
{doc_content}

Rules:
1. Whenever the document changes, ALWAYS call the update tool with the COMPLETE updated document.
2. Never just describe the update — always call the tool.
3. If the user wants to save, call the save tool with an appropriate filename.
4. If the user says goodbye or there is nothing more to do, call the save tool.
""")

    user_input = input("\nWhat would you like to do with the document? ").strip()
    if not user_input:
        user_input = "Done."

    print(f"\nUSER: {user_input}")
    user_message = HumanMessage(content=user_input)

    all_msg = [sys_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_msg)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using tools: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "messages": [user_message, response]
    }


def should_continue(state: AgentState) -> str:
    """Determine whether to continue or end the conversation."""
    messages = state["messages"]

    if not messages:
        return "continue"

    last_message = messages[-1]

    # If the last AI message has no tool calls, just end gracefully
    if isinstance(last_message, AIMessage):
        if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
            return "end"

    # Check the most recent ToolMessage for a save confirmation
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content_lower = message.content.lower()
            if "successfully saved" in content_lower and "document" in content_lower:
                return "end"
            break  # Only check the most recent ToolMessage

    return "continue"


def print_tool_results(messages):
    """Print the most recent tool result if present."""
    if not messages:
        return
    for message in reversed(messages[-3:]):
        if isinstance(message, ToolMessage):
            print(f"\nTOOL: {message.content}")
            break


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", doc_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "agent")

app = graph.compile()


def run_doc_agent():
    print("\n=== DRAFTER — Your AI Document Assistant ===")
    print("Type your instructions. Ask to save when done.\n")

    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_tool_results(step["messages"])

    print("\n=== DRAFTER SESSION ENDED ===")


if __name__ == "__main__":
    run_doc_agent()