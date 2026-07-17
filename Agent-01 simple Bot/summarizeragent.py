# from langgraph.graph import StateGraph,START,END
# from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama
# from typing import TypedDict,List
# class AgentState(TypedDict):
#     messages:List[HumanMessage]
    
# llm=ChatOllama(model="qwen2.5:7b",temperature=0)

# def summraizer(state:AgentState)->AgentState:
#     res4=llm.invoke(state["messages"])
#     print(f"\n Qwen: {res4.content}")
#     return state

# g6=StateGraph(AgentState)
# g6.add_node("summray",summraizer)
# g6.add_edge(START,"summray")
# g6.add_edge("summray",END)
# app3=g6.compile()


# input2=input("enter the text: ")


# app3.invoke({"messages":
#     [HumanMessage(
#         content=f"Summraize the following text:\n {input2}"
#         )]
#     })


from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

def summarizer(state: AgentState) -> AgentState:
    res4 = llm.invoke(state["messages"])
    print(f"\nQwen2.5:\n{res4.content}")
    return state

g6 = StateGraph(AgentState)
g6.add_node("summary", summarizer)
g6.add_edge(START, "summary")
g6.add_edge("summary", END)

app3 = g6.compile()

while True:

    print("\n===== AI Summarizer =====")
    print("1. One Line Summary")
    print("2. Five Bullet Points")
    print("3. Short Paragraph")
    print("4. Detailed Summary")
    print("5. Beginner Friendly Summary")
    print("Type 'exit' to quit")

    mode = input("\nChoose an option: ")

    if mode.lower() == "exit":
        break

    input2 = input("Enter the text: ")

    if mode == "1":
        prompt = f"Summarize the following text in exactly one sentence:\n{input2}"

    elif mode == "2":
        prompt = f"Summarize the following text in exactly five bullet points:\n{input2}"

    elif mode == "3":
        prompt = f"Summarize the following text in one short paragraph:\n{input2}"

    elif mode == "4":
        prompt = f"Provide a detailed summary of the following text:\n{input2}"

    elif mode == "5":
        prompt = f"Explain the following text in a beginner-friendly way:\n{input2}"

    else:
        print("Invalid choice!")
        continue

    app3.invoke({
        "messages": [
            HumanMessage(content=prompt)
        ]
    })

