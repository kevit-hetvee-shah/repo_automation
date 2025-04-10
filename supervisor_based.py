import json
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


# class Router(TypedDict):
#     next: Literal[*options]


class Router(TypedDict):
    next: Literal[*options]


class DebugCallbackHandler(BaseCallbackHandler):

    def on_llm_end(self, response, **kwargs):
        print("-----------------Response from LLM:-----------------")
        print(f"RESPONSE: {response}")
        print("******************************************************")


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


class SharedState(TypedDict):
    repo_name: Annotated[str, "Name of the repository."] = None
    extra_info: Annotated[str, "Additional information to be included in the README file."] = None
    file_path: Annotated[str, "Path of the file to be committed"] = None
    file_content: Annotated[str, "Content of the file to be committed"] = None
    commit_message: Annotated[Optional[str], "Message of the commit. Default is None"] = None
    branch_name: Annotated[Optional[str], "Name of the branch. Default is None"] = None
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created."
                        "Default is None")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True
    success_messages: Annotated[str, "Success messages from the tools."] = None
    messages: List = None


class CreateReadmeFileSchema(BaseModel):
    repo_name: Annotated[str, "Full Name of the repository."] = None
    extra_info: Annotated[str, "Additional information to be included in the README file."] = None


class CreateCommitSchema(BaseModel):
    repo_name: Annotated[str, "Full Name of the repository"]
    file_path: Annotated[str, "Path of the file to be committed"]
    file_content: Annotated[str, "Content of the file to be committed"]
    commit_message: Annotated[Optional[str], "Message of the commit. Default is None"] = None
    branch_name: Annotated[Optional[str], "Name of the branch. Default is None"] = None


class RepoCreationSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository"]
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created."
                        "Default is None")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True


@tool
def create_readme_file(readme_data: CreateReadmeFileSchema):
    """
    Tool to create the content of README file.
    """
    print(f"-----------README DATA-----------\n{readme_data} ")
    try:
        repo_name = readme_data.repo_name
        extra_info = readme_data.extra_info
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
        if extra_info:
            content += f"\n##Details:\n{extra_info}\n"
        print(
            f"-------------\nsuccess_messages: Successfully created the README file for the {repo_name} repository. \n file_content: {content}\n repo_name: {repo_name}\n---------")
        return {
            "success_messages": f"Successfully created the README file for the {repo_name} repository.",
            "file_content": content,
            "file_path": "./README.md",
            "commit_message": "Initial commit",
            "file_name": "README.md",
        }
    except Exception as e:
        print(f"Error in creating README file: {e}")
        breakpoint()
        return {
            "success_messages": f"Error in creating README file: {e}",
            "file_content": None,
            "file_path": "./README.md",
            "commit_message": "Initial commit",
            "file_name": "README.md",
        }


@tool
def create_commit(commit_data: CreateCommitSchema):
    """
    Tool to create a commit in a GitHub repository with given file path and content.
    """
    print(f"---------COMMIT DATA---------------\n{commit_data} ")
    try:
        # pass
        repo_name = commit_data.repo_name
        commit_message = commit_data.commit_message
        branch_name = commit_data.branch_name
        file_content = commit_data.file_content
        file_path = commit_data.file_path
        auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
        github_obj = Github(auth=auth)
        repo = github_obj.get_repo(repo_name)
        commit_obj = repo.create_file(path=file_path, content=file_content, message=commit_message, branch=branch_name)
        print(
            f"-------\ncommit_sha: f{commit_obj.get('commit').sha} \nfile_path: {file_path}\n file_content: {file_content}\n success_messages: Successfully created commit in {repo_name} at {branch_name} with file {file_path}\n----------------")

        return {
            "commit_sha": commit_obj.get('commit').sha,
            "file_path": file_path,
            "file_content": file_content,
            "success_messages": f"Successfully created commit in {repo_name} at {branch_name} with file {file_path}"
        }
    except Exception as e:
        print(f"Error in creating commit: {e}")
        breakpoint()
        return {
            "commit_sha": None,
            "file_path": None,
            "file_content": None,
            "success_messages": f"Error in creating commit: {e}"
        }


@tool
def create_github_repo(repo_data: RepoCreationSchema):
    """
    Tool to create a GitHub repository using given data.
    """
    print(f"---------------REPO DATA--------------\n{repo_data} ")
    try:
        repo_name = repo_data.repo_name
        organization_name = repo_data.organization_name
        description = repo_data.description
        private = repo_data.private
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
- extra_info: any extra information that is required to be included in the README file.

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
3. For creating a commit, you will get the values of following fields. Using these values, you will create a commit in the repository.
4. The repo_name, file_path, file_content, commit_message, branch_name fields will be required to create a commit and that will always be provided to you.

## field names:
- repo_name: full name of the repository to create commit in. 
- file_path: path where the file will be created. It will generally be "./file_name".
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

## field names:
- prompt: The prompt for which the code will be generated.

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

# Agents Description:
- create_repo_agent: This agent is responsible for creating a repository in the GitHub.
- create_readme_file_agent: This agent is responsible for creating the README file content for the given repository.
- create_commit_agent: This agent is responsible for creating a commit in the repository.
- generate_code_agent: This agent is responsible for generating the code in python language.

# NOTE:
- You must return the agent name in the response as it will be used by the agent to complete the task.
- If the agent response contains "FINAL ANSWER", then it means that the task is complete.
- When all tasks asked by user are complete, include "FINISH" as the final step.

# IMPORTANT
- If the agent's response contains "FINAL ANSWER" at beginning, middle or end, then it means that the agent task is complete. If any of the other agent can satisfy the user's query, call other agent else END the conversation.
- Analyze the response from agent including AIMessage and ToolMessage and decide what step to take next. 1. Call Agent or 2. End the conversation.
- If there's a need to call multiple agents, call them in order and define the sequence of agents to call.
"""


def readme_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_readme_file], prompt=readme_agent_prompt)
    print(f"---------- README AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    time.sleep(10)
    return Command(
        update={"messages": result['messages']},
        goto="supervisor",
    )


def create_commit_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm2, tools=[create_commit], prompt=create_commit_prompt)
    print(f"---------- COMMIT AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    time.sleep(10)
    return Command(
        update={"messages": result['messages']},
        goto="supervisor",
    )


def create_repo_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_github_repo], prompt=create_repo_prompt)
    print(f"----------REPO AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    time.sleep(10)
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)][-1]
    data = json.loads(tool_messages.content)
    print(f"----------DATA------------: \n{data}")
    return Command(
        update={
            "success_messages": data.get('success_messages'),
            "repo_name": data.get('repo_name'),
            "messages": result.get('messages'),
        },
        goto="supervisor",
    )


def generate_code_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[], prompt=generate_code_prompt)
    print(f"----------GENERATE CODE STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    time.sleep(10)
    return Command(
        update={"messages": result['messages']},
        goto="supervisor",
    )


# def excel_file_operations(state: SharedState) -> Command[Literal["supervisor"]]:
#     agent = create_react_agent(llm, tools=[], prompt=excel_file_operations_prompt)
#     print(f"----------GENERATE CODE STATE------------: \n{state}")
#     result = agent.invoke(state)
#     print(f"----------RESULT------------: \n{result}")
#     time.sleep(10)
#     return Command(
#         update={"messages": result['messages']},
#         goto="supervisor",
#     )


def supervisor_agent(state: SharedState) -> Command[
    Literal["create_repo", "create_readme_file", "create_commit", "generate_code", "__end__"]]:
    messages = [{"role": "system", "content": supervisor_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response['next']
    if goto == "FINISH":
        goto = END
    print(goto, "GOTO")
    time.sleep(10)
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
}, config=RunnableConfig(recursion_limit=16)):
    print(f"----------STREAM------------\n\n\n\n")
