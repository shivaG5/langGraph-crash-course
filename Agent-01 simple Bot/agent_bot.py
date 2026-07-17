from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama  
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
import langchain
load_dotenv()

class AgentState(TypedDict):
    messages:List[HumanMessage]# this is the human message that i am giving to the 

llm=ChatOllama(model="qwen2.5:7b",temperature=0)

def process(state:AgentState)->AgentState:
    response=llm.invoke(state["messages"]) #this is our question we want to ask that is in the message 
    print(f"\nQwen2.5: {response.content}")
    return state

#now creating the graph
graph=StateGraph(AgentState)
graph.add_node("process_node",process)
graph.add_edge(START,"process_node")
graph.add_edge("process_node",END)
Agent1=graph.compile()

user_input=input("Enter: ")
# Agent1.invoke({"messages":[HumanMessage(content=user_input)]}) #by these 2 part of code we can ask only 1 question and it ends there nothing more we can do so to make it real we keep the while condition
while user_input!="exit":
    Agent1.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input=input("Enter:")

#In thsis all we not created any memory so we complete in other parts