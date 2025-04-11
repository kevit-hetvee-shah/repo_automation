import json
from typing import TypedDict, Literal
from langgraph.graph import START, END
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.types import Command

from .agents.create_readme_content.CreateReadmeContentAgent import CreateReadmeContentAgent
from .agents.create_github_commit.CreateGitHubCommitAgent import CreateGitHubCommitAgent
from .agents.create_github_repo.CreateGitHubRepoAgent import CreateGitHubRepoAgent
from .agents.generate_code.GenerateCodeAgent import GenerateCodeAgent
from .agents.supervisor.SupervisorAgent import SupervisorAgent
from general.agents.utils.workflow import Workflow
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

callback_handler = OpenAICallbackHandler()

agent_nodes = ["create_repo", "create_readme_content", "create_commit", "generate_code"]

supervisor_next_nodes = agent_nodes + ["FINISH"]


class SharedState(TypedDict):
    """Shared state for the agent."""
    next: str
    messages: list[dict]


class Router(TypedDict):
    next: Literal[*supervisor_next_nodes]


def genereate_token_usage_str(cb):
    print(f"Total Tokens Used: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Successful Requests: {cb.successful_requests}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    return (f"Total Tokens Used: {cb.total_tokens}\nPrompt Tokens: {cb.prompt_tokens}\nCompletion Tokens:"
            f" {cb.completion_tokens}\nSuccessful Requests: {cb.successful_requests}\n"
            f"Total Cost (USD): ${cb.total_cost}")


def create_repo_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    result = CreateGitHubRepoAgent().create_agent().invoke(state, config={"callbacks":[callback_handler]})
    print(f"----------RESULT------------: \n{result}")
    token_data = genereate_token_usage_str(callback_handler)
    return_message = [HumanMessage(content=result['messages'][-1].content, name="create_repo")]
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    # if tool_messages:
    if 1 > 2:
        tool_message = tool_messages[-1]
        data = json.loads(tool_message.content)
        print(f"----------DATA------------: \n{data}")
        return Command(
            update={
                "success_messages": data.get('success_messages'),
                "repo_name": data.get('repo_name'),
                "messages": return_message,
            },
            goto="supervisor",
        )
    else:
        return Command(
            update={
                "messages": state['messages'] + [AIMessage(content=result["messages"][-1].content)]},
            goto="supervisor"
        )


def readme_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    result = CreateGitHubRepoAgent().create_agent().invoke(state, config={"callbacks":[callback_handler]})
    print(f"----------RESULT------------: \n{result}")
    token_data = genereate_token_usage_str(callback_handler)
    return_message = [HumanMessage(content=result['messages'][-1].content, name="create_repo")]
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    # if tool_messages:
    if 1 > 2:
        tool_message = tool_messages[-1]
        data = json.loads(tool_message.content)
        print(f"----------DATA------------: \n{data}")
        return Command(
            update={
                "success_messages": data.get('success_messages'),
                "repo_name": data.get('repo_name'),
                "messages": return_message,
            },
            goto="supervisor",
        )
    else:
        return Command(
            update={
                "messages": state['messages'] + [AIMessage(content=result["messages"][-1].content)]},
            goto="supervisor"
        )


def create_commit_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    result = CreateGitHubRepoAgent().create_agent().invoke(state, config={"callbacks":[callback_handler]})
    print(f"----------RESULT------------: \n{result}")
    token_data = genereate_token_usage_str(callback_handler)
    return_message = [HumanMessage(content=result['messages'][-1].content, name="create_repo")]
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    # if tool_messages:
    if 1 > 2:
        tool_message = tool_messages[-1]
        data = json.loads(tool_message.content)
        print(f"----------DATA------------: \n{data}")
        return Command(
            update={
                "success_messages": data.get('success_messages'),
                "repo_name": data.get('repo_name'),
                "messages": return_message,
            },
            goto="supervisor",
        )
    else:
        return Command(
            update={
                "messages": state['messages'] + [AIMessage(content=result["messages"][-1].content)]},
            goto="supervisor"
        )


def generate_code_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    result = CreateGitHubRepoAgent().create_agent().invoke(state, config={"callbacks":[callback_handler]})
    print(f"----------RESULT------------: \n{result}")
    token_data = genereate_token_usage_str(callback_handler)
    return_message = [HumanMessage(content=result['messages'][-1].content, name="create_repo")]
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    # if tool_messages:
    if 1 > 2:
        tool_message = tool_messages[-1]
        data = json.loads(tool_message.content)
        print(f"----------DATA------------: \n{data}")
        return Command(
            update={
                "success_messages": data.get('success_messages'),
                "repo_name": data.get('repo_name'),
                "messages": return_message,
            },
            goto="supervisor",
        )
    else:
        return Command(
            update={
                "messages": state['messages'] + [AIMessage(content=result["messages"][-1].content)]},
            goto="supervisor"
        )


def supervisor_node(state: SharedState) -> Command[Literal[*agent_nodes, "__end__"]]:
    supervisor_agent = SupervisorAgent()
    messages = [{"role": "system", "content": supervisor_agent.prompt.format(agent_nodes=agent_nodes)}] + state[
        "messages"]
    result = supervisor_agent.llm.with_structured_output(Router).invoke(messages)
    print(f"----------RESULT------------: \n{result}")
    token_data = genereate_token_usage_str(callback_handler)

    goto = result['next']
    if goto == "FINISH":
        goto = END
    print(goto, "GOTO")
    return Command(goto=goto, update={"next": goto})


workflow = Workflow(SharedState)
builder = workflow.generate_graph()
builder.add_node("supervisor", supervisor_node)
builder.add_node("create_repo", create_repo_agent)
builder.add_node("create_readme_content", readme_agent)
builder.add_node("create_commit", create_commit_agent)
builder.add_node("generate_code", generate_code_agent)

builder.add_edge(START, "supervisor")
graph = workflow.generate_compiled_graph()
query = input("Enter your query: ")
response = workflow.stream(query)
for i in response:
    print(f"-------STREAMING-----------\n{i}\n\n\n")
