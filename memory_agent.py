from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()
class AgentState(TypedDict):
    messages: List[Union[HumanMessage]] 


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def process(state: AgentState) -> AgentState:
    """"This node will solve the user input using the LLM."""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    # print(result["messages"])

    conversation_history = result["messages"]

    user_input = input("Enter your message: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")

    file.write("End of Discussion")

print("Conversation saved to logging.txt")