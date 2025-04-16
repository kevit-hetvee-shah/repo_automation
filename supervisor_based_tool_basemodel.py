import json
import re
import time
from github import UnknownObjectException
import os
from typing import Optional, List, TypedDict, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from github import Github, Auth
from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel
import openpyxl

load_dotenv()

members = ["create_repo", "create_readme_file", "create_commit", "generate_code"]

options = members + ["FINISH"]


class AgentSpec(TypedDict):
    id: str
    name: str


class Router(TypedDict):
    next: Literal[*options]
    agent_sequence: Annotated[List[str], "Sequence of the agents/nodes to be executed."]


class DebugCallbackHandler(BaseCallbackHandler):

    def on_llm_end(self, response, **kwargs):
        print(
            "--------------------------------------------------------------------------------------Response from LLM---------------------------------------------------------------------------------------------------")
        print(f"RESPONSE: {response}")
        print(
            "***********************************************************************************************************************************************************************************************************")


llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-flash",
    # model="gemini-2.0-flash-lite",
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048,
    timeout=None,
    max_retries=5,
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
    # callbacks=[DebugCallbackHandler()],
    verbose=True,
    # other params...
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    # model="gemini-2.0-flash-lite",
    # model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048,
    timeout=None,
    max_retries=5,
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
    callbacks=[DebugCallbackHandler()],
    verbose=True,
    # other params...
)


class ToolData(BaseModel):
    repo_name: Annotated[str, "Name of the repository."] = None
    file_path: Annotated[str, "Path of the file"] = None
    file_content: Annotated[str, "Content of the file."] = None
    commit_message: Annotated[Optional[str], "Message of the commit"] = None
    branch_name: Annotated[Optional[str], "Name of the branch. Default is None"] = "main"
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created."
                        "Default is None")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True
    success_messages: Annotated[str, "Success messages from the tools."] = None


class SharedState(TypedDict):
    tool_data: ToolData
    messages: List = None


@tool
def create_readme_file(tool_data: ToolData):
    """
    Tool to create the content of README file.
    """
    print(
        f"----------------------------------------------------------------------------------------------------README DATA---------------------------------------------------------------------------------\n{tool_data} ")
    try:
        repo_name = tool_data.repo_name
        content = f"""
        # {repo_name}\nThis is the README file for the {repo_name} repository.

            ## Prerequisite: \n- Python 3.8+

            ## Installation:
            \n```bash\n
            pip3 install -r requirements.txt\n
            ```

            ## Run code: \n
            ```bash\n
            python3 main.py\n
            ```
            """
        print(
            f"-------------\nsuccess_messages: Successfully created the README file for the {repo_name} repository. \n with readme file content \n repo_name: {repo_name}\n---------")
        return {
            "success_messages": f"Successfully created the README file for the {repo_name} repository.",
            "file_content": content,
            "file_path": "./README.md",
            "commit_message": "Initial commit",
        }
    except Exception as e:
        print(f"Error in creating README file: {e}")
        breakpoint()
        return {
            "success_messages": f"Error in creating README file: {e}",
            "file_content": None,
            "file_path": "./README.md",
            "commit_message": "Initial commit",
        }


@tool
def create_commit(tool_data: ToolData):
    """
    Tool to create a commit in a GitHub repository with given file path and content.
    """
    print(
        f"--------------------------------------------------------------------------------------------COMMIT DATA------------------------------------------------------------------------------------------\n{tool_data} ")
    try:
        repo_name = tool_data.repo_name
        commit_message = tool_data.commit_message
        branch_name = tool_data.branch_name
        file_content = tool_data.file_content
        file_path = tool_data.file_path
        auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
        github_obj = Github(auth=auth)
        repo = github_obj.get_repo(repo_name)
        commit_obj = repo.create_file(path=file_path, content=file_content, message=commit_message, branch=branch_name)
        print(
            f"-------\ncommit_sha: f{commit_obj.get('commit').sha} \nfile_path: {file_path}\n file_content: {file_content}\n success_messages: Successfully created commit in {repo_name} at {branch_name} with file {file_path}\n----------------")

        return {
            "success_messages": f"Successfully created commit with sha {commit_obj.get('commit').sha} in {repo_name} at {branch_name} with file {file_path}"
        }
    except Exception as e:
        print(f"Error in creating commit: {e}")
        breakpoint()
        return {
            "commit_sha": None,
            "success_messages": f"Error in creating commit: {e}"
        }


@tool
def create_github_repo(tool_data: ToolData):
    """
    Tool to create a GitHub repository using given data.
    """
    print(
        f"---------------------------------------------------------------------------------------REPO DATA----------------------------------------------------------------------------------------------\n{tool_data} ")
    try:
        repo_name = tool_data.repo_name
        organization_name = tool_data.organization_name
        description = tool_data.description
        private = tool_data.private
        auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
        github_obj = Github(auth=auth)
        if organization_name:
            organization = github_obj.get_organization(organization_name)
            try:
                existing_repo = organization.get_repo(repo_name)
                repo = existing_repo
            except UnknownObjectException as e:
                new_repo = organization.create_repo(
                    name=repo_name,
                    allow_rebase_merge=True,
                    description=description,
                    private=private,
                )
                repo = new_repo
        else:
            user = github_obj.get_user()
            try:
                existing_repo = user.get_repo(repo_name)
                repo = existing_repo
            except UnknownObjectException as e:
                new_repo = user.create_repo(name=repo_name,
                                            description=description if description else "Repository created by langgraph",
                                            private=private)
                repo = new_repo
        github_obj.close()

        print(
            f"--------\nMessage: Successfully created the {repo.full_name} repository in github. \nrepo_name: {repo.full_name}\n-------------")
        return {
            "success_messages": f"Successfully created the {repo.full_name} repository in github.",
            "repo_name": repo.full_name,
        }

    except Exception as e:
        print(f"Error in creating repository: {e}")
        breakpoint()
        return {
            "success_messages": f"Error in creating repository: {e}",
            "repo_name": None,
        }


readme_agent_prompt = """# CreateReadmeContentAgent Instructions

You are an agent responsible for creating the README file content with given data.
Your responses should only address the creation of the content for README file. 
Do not include data from any external context. Operate as a standalone assistant focused solely on this task.

# Primary Instructions
1. You must use the create_readme_file tool to create the content of the README file.

## field names:
- repo_name: name of the repository and its required.

# NOTE:
- Dont ask any questions to the user. Just create the content of the README file with the given instructions.
- Just use tools and complete the task that can be completed by you and return the response. 
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.
"""

create_commit_prompt = """# CreateCommitAgent Instructions

You are an agent responsible for creating a commit with the given data in the repository.

# Primary Instructions
1. You must use the create_commit tool to create the commit.
2. Your role is to create a commit with the given data in the repository.
3. The repo_name, file_path, file_content, commit_message, branch_name fields will be required to create a commit. All these values will be provided to you. Use those provided values to create a commit.

## field names:
- repo_name: full name of the repository to create commit in. 
- file_path: path where the file will be created.
- file_content: content of the file to be committed.
- commit_message: message for the commit.
- branch_name: name of the branch where the commit will be made.

# NOTE:
- Dont ask any questions to the user. Just create the commit with the given instructions.
- Just use the given tools and complete the task that can be completed by you and return the response.
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.
- If you are unable to create a commit, you should return "Error in creating commit: <error message>".

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.

"""

create_repo_prompt = """# CreateRepoAgent Instructions

You are an agent responsible for creating a repository in the GitHub. 

# Primary Instructions
1. You must use the create_repo_tool to create the repository.

## field names:
- repo_name: name of the repository.
- description: description of the repository.
- organization_name: name of the organization.
- private: whether the repository is private or not.

# NOTE:
- You should not ask any questions to the user. Just simple create the repository with the given instructions.
- Just use tools and complete the task that can be completed by you and return the response. Other tools will do the rest of the work.
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.
"""

generate_code_prompt = """## GenerateCodeAgent Instructions

You are an agent responsible for generating the code. 

# Primary Instructions
1. You must generate code in python language only.
2. The code must not have any errors.
3. The code must be well formatted.
4. The code must be well commented.
5. The code must contain at least 1 example. 
6 If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.
7. You must return the return fields in the JSON format.

## field names:
- prompt: The prompt for which the code will be generated.

## return fields:
- file_name: Name of the file based on user's query
- file_content: The actual python code
- commit_message: The message explaining using query

# NOTE:
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.
"""

supervisor_prompt = f"""# Supervisor Instructions


# Primary Instructions
- You are an agent responsible for supervising the other agents.
- Your role is understand the user input and decide which agent to use.
- You have these agents with you {members}
- Each agent of highly capable of doing a specific task.
- You must use the agent that is best suited for the task.
- You should first create a sequence of agents in order to complete the task.

# Agents Description:
- create_repo: This agent is responsible for creating a repository in the GitHub.
- create_readme_file: This agent is responsible for creating the README file content for the given repository.
- create_commit: This agent is responsible for creating a commit in the repository. You should use this agent when anything needs to be commited to repository.
- generate_code: This agent is responsible for generating the code in python language.

# NOTE:
- You must return the agent name in the response as it will be used by the agent to complete the task.
- Use create_commit to create commit
- If the agent response contains "FINAL ANSWER", then it means that the task is complete.
- When all tasks asked by user are complete, include "FINISH" as the final step.
- You should create a sequence of agents in order to complete the task.

# Your Responsibilities:
        1. Understand User Requests: Given the user's input, identify ALL workers needed to handle it and the proper 
        sequence they should be executed in.
        2. Routing Logic: Assign the appropriate workers from the available options [{", ".join(members)}].
           When all tasks are complete, include "FINISH" as the final step.
        3. Response Handling:
           - Structured Output: Ensure all responses are in JSON format with the keys:
             - `tool_sequence`: An ordered array of worker names to execute in sequence
             - For each step, include any specific instructions or parameters needed
        4. Finalization: Once all tools have been executed, add "FINISH" as the final element in the tool_sequence.

#Communication Style:
        - Keep responses clear, structured, and informative.
        - For multi-tool sequences, explain the plan for executing tasks in order.
        - Ensure all questions are concise and easy to understand.
        - When the conversation is complete, always include "FINISH" as the final step.

# IMPORTANT
- If the agent's response contains "FINAL ANSWER" at beginning, middle or end, then it means that the agent task is complete. If any of the other agent can satisfy the user's query, call other agent else END the conversation.
- Analyze the response from agent including AIMessage and ToolMessage and decide what step to take next. 1. Call Agent or 2. End the conversation.
- If there's a need to call multiple agents, call them in order and define the sequence of agents to call.
- You should create a sequence of agents in order to complete the task.
- If a query requires multiple tools, you MUST identify all needed tools and their proper execution 
        order. Do not simply route to one tool when multiple are needed.
"""


def readme_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_readme_file], prompt=readme_agent_prompt)
    print(
        f"---------------------------------------------------------------------------------------------------------- README AGENT STATE------------------------------------------------------------------------: \n{state}")
    result = agent.invoke(state)
    print(
        f"---------------------------------------------------------------------------------------------------------RESULT-------------------------------------------------------------------------------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        data = json.loads(tool_message.content)
        updated_state = {
            "success_messages": data.get('success_messages'),
            "messages": result.get('messages'),
            "file_name": data.get('file_name'),
            "file_content": data.get('file_content'),
            "file_path": data.get('file_path'),
            "commit_message": data.get('commit_message'),
        }
        return Command(
            update={
                "tool_data": updated_state,
                "messages": result.get('messages'),
            },
            goto="supervisor",
        )
    else:
        return Command(
            update={
                "messages": state['messages']},
            goto="supervisor"
        )


def create_commit_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_commit], prompt=create_commit_prompt)
    print(
        f"------------------------------------------------------------------------------------------- COMMIT AGENT STATE-------------------------------------------------------------------------------------: \n{state}")
    result = agent.invoke(state)
    print(
        f"------------------------------------------------------------------------------------------------RESULT---------------------------------------------------------------------------------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            updated_state = {
                "success_messages": data.get('success_messages'),
            }
            return Command(
                update={
                    "tool_data": updated_state,
                    "messages": result.get('messages'),
                },
                goto="supervisor",
            )
        except Exception as e:
            print(f"Exception: {e}")
            breakpoint()
            return Command(
                update={
                    "messages": result.get('messages'),
                },
                goto="supervisor",
            )
    else:
        return Command(
            update={
                "messages": state['messages']},
            goto="supervisor"
        )


def create_repo_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_github_repo], prompt=create_repo_prompt)
    print(
        f"---------------------------------------------------------------------------------------------REPO AGENT STATE-------------------------------------------------------------------------------------: \n{state}")
    result = agent.invoke(state)
    print(
        f"-----------------------------------------------------------------------------------------------RESULT--------------------------------------------------------------------------------------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        data = json.loads(tool_message.content)
        updated_state = {
            "success_messages": data.get('success_messages'),
            "repo_name": data.get('repo_name'),
            "messages": result.get('messages'),
        }
        return Command(
            update={
                "tool_data": updated_state,
                "messages": result.get('messages'),
            },
            goto="supervisor",
        )
    else:
        return Command(
            update={
                "messages": state['messages']},
            goto="supervisor"
        )


def generate_code_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[], prompt=generate_code_prompt)
    print(
        f"-----------------------------------------------------------------------------------GENERATE CODE STATE------------------------------------------------------------------------------------------: \n{state}")
    result = agent.invoke(state)
    print(
        f"--------------------------------------------------------------------------------------RESULT---------------------------------------------------------------------------------------------------: \n{result}")
    # file_content = [i.content for i in result['messages'] if type(i) == AIMessage and i.content != ""][0]
    # if "json" in file_content:
    # cleaned = re.sub(r"^```json\n|```$", "", file_content.strip(), flags=re.MULTILINE).replace("FINAL ANSWER", "")
    # breakpoint()
    # json_data = json.loads(cleaned)

    # updated_state = {
    #     "file_content": json_data.get('file_content'),
    #     "file_path": f"./{json_data.get('file_name')}",
    #     "commit_message": json_data.get('commit_message')
    # }
    updated_state = {
        "file_content": "CODE",
        "file_path": "./code.py",
        "commit_message": "test_commit"
    }
    return Command(
        update={
            "tool_data": updated_state,
            "messages": result.get('messages'),
        },
        goto="supervisor",
    )


def supervisor_agent(state: SharedState) -> Command[
    Literal["create_repo", "create_readme_file", "create_commit", "generate_code", "__end__"]]:
    messages = [{"role": "system", "content": supervisor_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    breakpoint()
    goto = response['next']
    if goto == "FINISH":
        goto = END
    print(goto, "GOTO")
    return Command(goto=goto, update={"next": goto})


builder = StateGraph(SharedState)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("create_repo", create_repo_agent)
builder.add_node("create_readme_file", readme_agent)
builder.add_node("create_commit", create_commit_agent)
builder.add_node("generate_code", generate_code_agent)

builder.add_edge(START, "supervisor")

graph = builder.compile()
query = input("Enter the query here: ")
for q in graph.stream({
    "messages": [
        HumanMessage(role="user", content=query)
    ],
    "tool_data": {}
}, config=RunnableConfig(recursion_limit=16)):
    print(
        f"----------------------------------------------------------------------------------------------------------STREAM------------------------------------------------------------------------------------\n\n\n\n{q}")
