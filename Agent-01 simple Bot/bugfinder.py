from typing import TypedDict,List
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages:List[HumanMessage]
    

llm=ChatOllama(model="qwen2.5:7b",temperature=0)

def bugfinder(state:AgentState)->AgentState:
    res4=llm.invoke(state['messages'])#This must be same as the above class AgentState
        
    return state

g7=StateGraph(AgentState)
g7.add_node("Bugfinder",bugfinder)
g7.add_edge(START,"Bugfinder")
g7.add_edge("Bugfinder",END)
AgentBug=g7.compile()

uinput=input("Enter the code: ")
prompt=f""" Analyze the following code. 
    return your answer in thsi format:
    1.errors found
    2.Reason
    3.Corrected code
    4.Improvement Suggestions
    code:
    {uinput}
    """
AgentBug.invoke({
    "messages":[
        HumanMessage(content=prompt)
    ]
})