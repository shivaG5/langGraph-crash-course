# from typing import TypedDict,List
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph,START,END
# from langchain_core.messages import HumanMessage
# from dotenv import load_dotenv
# load_dotenv()

# class AgentState(TypedDict):
#     msg1:List[HumanMessage]
    
# llm2=ChatOllama(model="qwen2.5:7b",temperature=0)

# def process1(state:AgentState)->AgentState:
#     response=llm2.invoke(state["msg1"])#model invoking
#     print(f"\nQwen-2.5: ,{response.content}")
#     return state


# g1=StateGraph(AgentState)
# g1.add_node("Process",process1)
# g1.add_edge(START,"Process")
# g1.add_edge("Process",END)
# PA1=g1.compile()

# uinput=input("Enter: ")
# PA1.invoke({"msg1":[HumanMessage(content=uinput)]})



from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    msg2:List[HumanMessage]

model1=ChatOllama(model="qwen2.5:7b",temperature=0)
def process2(state:AgentState)->AgentState:
    res2=model1.invoke(state["msg2"])
    print(f"\nQwen: ,{res2.content}")
    return state

g2=StateGraph(AgentState)
g2.add_node("new_process",process2)
g2.add_edge(START,"new_process")
g2.add_edge("new_process",END)
Ag2=g2.compile()

uin=input("Enter the Question: ")
Ag2.invoke({"msg2":[HumanMessage(content=uin)]})
















# from typing import TypedDict,List
# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama  
# from langgraph.graph import StateGraph,START,END
# from dotenv import load_dotenv
# import langchain
# load_dotenv()

# class AgentState(TypedDict):
#     messages:List[HumanMessage]# this is the human message that i am giving to the 

# llm=ChatOllama(model="qwen2.5:7b",temperature=0)

# def process(state:AgentState)->AgentState:
#     response=llm.invoke(state["messages"]) #this is our question we want to ask that is in the message 
#     print(f"\nQwen2.5: {response.content}")
#     return state

# #now creating the graph
# graph=StateGraph(AgentState)
# graph.add_node("process_node",process)
# graph.add_edge(START,"process_node")
# graph.add_edge("process_node",END)
# Agent1=graph.compile()

# user_input=input("Enter: ")
# # Agent1.invoke({"messages":[HumanMessage(content=user_input)]}) #by these 2 part of code we can ask only 1 question and it ends there nothing more we can do so to make it real we keep the while condition
# while user_input!="exit":
#     Agent1.invoke({"messages":[HumanMessage(content=user_input)]})
#     user_input=input("Enter:")

#In thsis all we not created any memory so we complete in other parts