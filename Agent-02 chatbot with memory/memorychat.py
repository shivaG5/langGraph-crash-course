import os
from typing import TypedDict,List,Union
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]# by this union we can store either Humanmessage or the AImessage
    #aimessage:List[AIMessage]this is one way to use it or u can use first one 
    
llm=ChatOllama(model="qwen2.5:7b",temperature=0)

def process(state:AgentState)->AgentState:
    """This node will solve the requested input"""
    response=llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))    
    #response.content->The content of the response that we declared on tip 
    print(f"\n AI: {response.content}")
    print("Current State: ",state["messages"])#it prints the current state of now 
    return state


graph=StateGraph(AgentState)
graph.add_node("Process",process)
graph.add_edge(START,"Process")
graph.add_edge("Process",END)
AgentM=graph.compile()

conservation_history=[] 
user_input=input("Enter: ")
while user_input!="exit":
    conservation_history.append(HumanMessage(content=user_input)) #adding upto the conservation history List we also give the HumanMessage up here that is userinput and why content-It is a parameter in the human message
    result=AgentM.invoke({"messages":conservation_history})#assigning an variable to it ("result")and invoke agent to get conservation history(CH) and store it in messgae by this the entire CH is sent not the only new asked one ,it and along with all old chat will go with it 
    #print(result["messages"])
    conservation_history=result["messages"] #replace the conservation history completely with wipe it with the result message 
    user_input=input("Enter: ")

#the  main problem here is that when i do exit and run program it will not remember nothing and if we ask our name it will show something like this -I'm sorry i don't have access to personal_data..." like this cause the data is wiped away just like that it will not have the memory stored fully like gpt,claude etc.The solution is to store in an large database and like we did for rag we store it in an VDB now we store it in an simple text file why cause it is easy now as we are just doing an  protoyping we use text file now code for the text file 

with open("logging.txt","w",encoding="utf-8")as file: #file created named as logging and it is in write mode
    file.write("Your Conservation Log: \n")#it is conservation_log  
    for message in conservation_history:#for every msg in CH,the CH store both Human and ai messages that is outside grapha and state is inside the graph and conservation hostory is an duplicate version of the state caused we have appeneded it in 2 lines in the while loop(1,3)
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conservation")
    
print("Conservation saved to the logging.txt") #this is saving the text is not roboust way but the fast way than better


# [HumanMessage(content='hello how are you', additional_kwargs={}, response_metadata={}), AIMessage(content="Hello! I'm an AI assistant created by Alibaba Cloud, so I don't have personal feelings or experiences. But I'm here and ready to help you with any questions or tasks you might have! How can I assist you today?", additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='my name is steven i need some help with math homework', additional_kwargs={}, response_metadata={}), AIMessage(content="Hello Steven! I'd be happy to help you with your math homework. What specific topic are you working on, and do you have any particular problems you're struggling with? Whether it's algebra, geometry, calculus, or something else, just let me know, and we can get started!", additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='what is 998+500', additional_kwargs={}, response_metadata={}), AIMessage(content="Sure thing, Steven! Let's add those numbers together:\n\n\\[ 998 + 500 = 1498 \\]\n\nSo, the answer is \\( 1498 \\). If you have any more questions or need help with anything else, feel free to ask!", additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='tell me an joke on math', additional_kwargs={}, response_metadata={}), AIMessage(content="Of course! Here's a light-hearted math joke for you:\n\nWhy did the mathematician refuse to go to the beach?\n\nBecause he didn't want to deal with tangent lines and secant waves! \n\nI hope that brought a smile to your face! Do you have any other questions or need help with more math problems?", additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[])] right now what u see these are tokens and these will especially input tokens eat away the cost and money so how to solve this easy one -within the code,write this-when number of human messages exceeds 5 we remove the first message in our history why first not latetst well latest is most relevant ,and first one may not be needed
        