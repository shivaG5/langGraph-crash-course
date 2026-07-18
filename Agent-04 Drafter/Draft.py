from typing import TypedDict,Annotated,Sequence
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

#Global variable that it can store the doc why we did it-It is to actually pass in an a state in tools and in langgraph we did it using injected state->What it does is our tools whenever updates are made it will automatically update the global variable

doc_content=""

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

#2tools-update, save tool   
@tool
def update(content:str)->str: #content-give by llm 
    """Updates the document with the provided contents."""
    global doc_content
    doc_content=content
    return f"Document has been  updated successfully! The current content is:\n{doc_content}"

@tool
def save(filename:str)->str: #to give us the filename from the llm direct
    """Save the current document  to a text file and finish the process
    Args:
     filename:Name for the text file.
    """
    #if it is not in text just convert to text
    global doc_content
    
    if not filename.endswith('.txt'):
        filename=f"{filename}.txt"
        
    try:
        with open(filename,"w") as file:
            file.write(doc_content)
        print(f"\n document has been saved to : {filename}")
        return f"Document has successfully saved to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
    
tools=[update,save]
model=ChatOllama(model="qwen2.5:7b",temperature=0).bind_tools(tools)

#now agent is initalizing

def doc_agent(state:AgentState)->AgentState:
    sys_prompt=SystemMessage(content=f""" You are Drafter.

    Current document:
    {doc_content}

    Rules:
    1. Whenever the document changes, ALWAYS call the update tool.
    2. Pass the COMPLETE updated document to the update tool.
    3. Never just describe the update.
    4. If the user wants to save, call the save tool.
   """)
    #The case when we did not get any doc
    if not state["messages"]:
        user_input="I'm ready to help you update a document.What would you like to create?"
        user_message=HumanMessage(content=user_input)#it collects user_input
        
     #now if we already kept an doc or updating drafting it .well to do that we need this else statement
    else:
        user_input=input("\n What would you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message=HumanMessage(content=user_input)
        
    all_msg=[sys_prompt]+list(state["messages"])+[user_message]
    response=model.invoke(all_msg)
    print(f"\n AI: {response.content}")
    if hasattr(response,"tool_calls") and response.tool_calls:
        print(f"Using tools: {[tc['name']for tc in response.tool_calls]}")
        
    return {
        "messages":
            [
                user_message,
                response
            ]
        }

#Now let's create an conditional edge

def should_continue(state:AgentState)->str:
    """Determine if we should continue or end the conservation"""
    messages=state["messages"]
    if not messages:
        return "continue"
    
    #This looks  for the most recent tool message...
    for message in reversed(messages):
        #... and checks if this is a ToolMessage resulting from save
        if(isinstance(message,ToolMessage)and
           "saved" in message.content.lower()and
           "document" in message.content.lower()):
            return "end" #goes to the end edge which leads to the endpoint we will see whether thsi tool used the saved tool or not we have 2 conditions one is to save the doc then save tool ->end and if update then go to agent update->Agent(run again)
        
    return "continue" #use it by defalut->so it can go to the next tool that is update tool aka it goes to continue edge


def print_msg(messages):
    """function I made to print the message in a more readable format"""
    if not messages:
        return 
    
    for message in messages[-3:]:
        if isinstance(message, HumanMessage):
           print("\nUSER:", message.content)

        elif isinstance(message, AIMessage):
             print("\nAI:", message.content)

        elif isinstance(message, ToolMessage):
           print("\nTOOL:", message.content)
  
  
#Now doing the graph part-
graph=StateGraph(AgentState)
graph.add_node("agent",doc_agent)
graph.add_node("tools",ToolNode(tools))
graph.set_entry_point("agent")
# graph.add_edge("agent","tools")#here the two lines that we want to connect the edges both   
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue":"tools",#to telling it where to go next see img
        "end":END,#the end when there is nothing
    },
)      
graph.add_edge("tools","agent")      
app=graph.compile()

def run_doc_agent():
    print("\n===DRAFTER===")
    state={"messages":[]}
    for step in app.stream(state,stream_mode="values"):
        if"messages" in step:
            print_msg(step["messages"])
            
    print("\n === DRAFTER====")
    
if __name__ == "__main__":
    run_doc_agent()