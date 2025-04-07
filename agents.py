from typing import TypedDict, Annotated, List, Literal

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .commit_to_github import CommitToGitHubAgent
from .create_repo import CreateRepoAgent
from .generate_readme_file_content import CreateReadmeContentAgent
from .generate_code import GenerateCode

class State(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    iteration: int
    expected_result: str
    feedback: str
    agent_outputs: Annotated[List[AIMessage], add_messages]
    evaluation: float
    continue_loop: bool  # Flag to continue the loop

builder = StateGraph(State)
agents = {
    "generate_code": GenerateCode,
    "create_repo": CreateRepoAgent,
    "commit_to_github": CommitToGitHubAgent,
    "create_readme_file_content": CreateReadmeContentAgent,
}

for node_name, agent in agents.items():
    breakpoint()
    builder.add_node(node_name, agent)

builder.add_edge(START, "create_repo")


def next_node(state: State) -> Literal["supervisor", END]:
    if state['continue_loop']:
        return "supervisor"
    else:
        return END


graph = builder.compile()

q = "What is the code for fibonacci series."
initial_state = State(
    messages=[HumanMessage(role="user", content=q)],
)

# Execute the graph
final_state = graph.invoke(initial_state, {"recursion_limit": 5})
