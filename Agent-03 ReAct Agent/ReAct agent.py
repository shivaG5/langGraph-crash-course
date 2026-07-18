#ReAct Reasosing and Acting Agent-#

from typing import Annotated,Sequence,TypedDict 
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage 
from langchain_core.messages import ToolMessage 
from langchain_core.messages import SystemMessage 
from langchain_core.tools import tool
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    
    
@tool
def add(a:int,b:int):
    """This is an addition function that adds 2 numbers"""
    return a+b

@tool
def sub(a:int,b:int):
    """This is an subtraction function that subtracts 2 numbers"""
    return a-b

@tool
def mul(a:int,b:int):
    """This is an Multiplication function that multiply 2 numbers"""
    return a*b


#How to connect to llm - simple use the langgraph
tools=[add,sub,mul]

model=ChatOllama(model="qwen2.5:7b",temperature=0).bind_tools(tools) #Now how to tell it u have this tool use it -we use.bind_tools(tools)

#Now to create an node that act as an agent in the graph
def model_call(state:AgentState)->AgentState:
    system_prompt=SystemMessage(content="You are an Ai assistant,please  answer the query with your best ability")
    response=model.invoke([system_prompt]+state["messages"])#here the message is in the query form that is humanmsg
    return {"messages":[response]} #it is just another way of telling the updated state,we uodate msg with response here add_message will handle the appending

#now we need an conditional_edges
def should_continue(state:AgentState)->str:
    messages=state["messages"]
    print(type(messages))
    print(messages)
    print(" = "*50)
    last_message=messages[-1] #what are doing here is getting the last msg and seeing where it required tools are not  then go to continue edge an do it oe else then end 
    print(type(last_message))
    print(last_message)
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue" #this is an edge we define it later 
    

g2=StateGraph(AgentState)
g2.add_node("our_agent",model_call)

tool_node=ToolNode(tools=tools)
g2.add_node("tools",tool_node)

g2.set_entry_point("our_agent")
g2.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    },
)

g2.add_edge("tools","our_agent")#this connecting back to the agent and we will connect it to the back send and tell to the agent
app=g2.compile()

def print_stream(stream):
    for s in stream:
        message=s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else: 
            message.pretty_print() #This for the correct output format
            
        
inputs={"messages":[("user","Add 9+10,and then sub by 5 and mul by 2,And tell an spiderman joke")]}
print_stream(app.stream(inputs,stream_mode="values"))
            