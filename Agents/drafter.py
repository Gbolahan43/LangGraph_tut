from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os


load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool

def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document updated successfully! The current content is:\n{document_content}."

@tool
def save(filename: str) -> str:
    """Saves the current document content to a text file and finish the drafting process.

    Better to use .txt extension for the filename.
    provide file name if not given .txt will be added automatically.
    
    Args:
    filename (str): The name of the file to save the document content to.
    """

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    
    
    try:
        with open(filename, 'w') as file:
            file.write(document_content) 
        print(f"\nDocument saved successfully to: {filename}.")
        return f"Document saved successfully as {filename}."
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"

tools = [update, save]

# model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0).bind_tools(tools)

# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.7
# ).bind_tools(tools)

model = ChatGroq(
    model="llama-3.3-70b-versatile",  # Very fast and capable
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

def agent_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    You are a document drafting AI assistant. Use the provided tools to update and save the document as per user instructions.
    - If the user wants to update or modify content, use the 'update' tool with the complte updated content.
    - If the user wants to save and finish, use the 'save' tool.
    -Make sure to always to always show the current content of the document when updating.

    The current document content is:
    {document_content}                              
                                                
    """)

    if not state["messages"]:
        user_input = "Hello, I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
        # state["messages"] = [user_message]

    else:
        user_input = input("\n What would you like to do with the document? ")
        print(f"\nUser: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

 
def should_continue(state: AgentState) -> str:
    """Decides whether to continue or end the conversation """

    messages = state["messages"]

    if not messages:
        return "continue"
    for message in reversed(messages):
        # .. check for ToolMessage indicating document saved
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
        
    return "continue" 
            

    # if not isinstance(last_message, ToolMessage) or last_message.tool_name == "save":
    #     return "end"
    # else:
    #     return "continue"

def print_messages(messages):
    """Function to print the conversation messages in a readable format."""

    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n[Tool Result: {message.content}]")

graph = StateGraph(AgentState) 
graph.add_node("agent", agent_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node) 
graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_drafter():
    print("\n ==== Welcome to the Document Drafter AI Assistant!====\n")
    state: AgentState = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ==== Document Drafting Session Ended. ==== \n")

if __name__ == "__main__":
    run_drafter()
 