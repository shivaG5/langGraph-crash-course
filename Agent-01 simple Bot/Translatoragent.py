from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages:List[HumanMessage]
    
llm=ChatOllama(model="qwen2.5:7b",temperature=0)
    
def translator(state:AgentState)->AgentState:
    res3=llm.invoke(state["messages"])
    print(f"\n Qwen2.5: ,{res3.content}")
    return state

g5=StateGraph(AgentState) 
g5.add_node("Translator",translator) 
g5.add_edge(START,"Translator")
g5.add_edge("Translator",END) 
Tagent=g5.compile()

language=input("Enter the language: ")
uinput=input("Enter the text: ")

while uinput!="exit":
    Tagent.invoke({
        "messages":[
            HumanMessage(content=f"Translate this into {language}:\n{uinput}")
            ]})
     
    uinput=input("Enter the text: ")