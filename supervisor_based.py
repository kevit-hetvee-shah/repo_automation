import json

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

load_dotenv()


class DebugCallbackHandler(BaseCallbackHandler):

    def on_llm_end(self, response, **kwargs):
        print("-----------------Response from LLM:-----------------")
        print(f"RESPONSE: {response}")
        print("******************************************************")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
    # callbacks=[DebugCallbackHandler()],
    verbose=True,
    # other params...
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
    # callbacks=[DebugCallbackHandler()],
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
        repo_name = f"kevit-hetvee-shah/{commit_data.repo_name}"
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

members = ["create_repo", "create_readme_file", "create_commit", "generate_code"]

options = members + ["FINISH"]

class Router(TypedDict):
    next: Literal[*options]

readme_agent_prompt = """# CreateReadmeContentAgent Instructions

You are an agent responsible for creating the README file content for the given repository.
Your responses should only address the creation of the content for README file. 
Do not include data from any external context. Operate as a standalone assistant focused solely on this task.

# Primary Instructions
1. You must use the create_readme_file tool to create the content of the README file.
2. Make sure the content is in markdown format.

## field names:
- repo_name: name of the repository and its required.
- extra_info: any extra information that is required to be included in the README file.

# NOTE:
- Dont ask any questions to the user. Just create the content of the README file with the given instructions.
- Just use tools and complete the task that can be completed by you and return the response. Other tools will do the rest of the work.

# Example Output:
Make sure the JSON object is correctly formatted and contains no placeholder values (e.g., "unknown").

{
    "repo_name": "abc",
    "file_path": "./file",
    "file_content": "This is the readme file content",
}
"""

create_commit_prompt = """# CreateCommitAgent Instructions

You are an agent responsible for creating a commit in the repository.

# Primary Instructions
1. You must use the create_commit tool to create the commit.


## field names:
- repo_name: full name of the repository to create commit in. 
- file_path: path where the file will be created. It will generally be "./file_name".
- file_content: content of the file to be committed.
- commit_message: message for the commit.
- branch_name: name of the branch where the commit will be made.

# NOTE:
- Dont ask any questions to the user. Just create the commit with the given instructions.
- If you cant fulfil the task, just return the response as it is.
- Just use tools and complete the task that can be completed by you and return the response. Other tools will do the rest of the work.
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
"""

generate_code_prompt = """## GenerateCodeAgent Instructions

You are an agent responsible for generating the code. 

You must return following data:
- file_name: Name of the file. 
- file_path: Path of the file. Path will always be "./file_name". 
- file_content: The content generated.
- branch_name: The name of the branch where the file will be created.
- commit_message: The commit message for the commit. 

# Primary Instructions
1. You must generate code in python language only.
2. The code must not have any errors.
3. The code must be well formatted.
4. The code must be well commented.
5. The code must contain at least 1 example. 

## field names:
- prompt: The prompt for which the code will be generated.

# NOTE:
- You must return the file_name, file_path, file_content, branch_name and commit_message in the response as it will be used by create_commit_agent to create commit of generated files.
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
When you are done, respond with FINISH.
"""

def readme_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_readme_file], prompt=readme_agent_prompt)
    print(f"---------- README AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)][-1]
    data = json.loads(tool_messages.content)
    print(f"----------DATA------------: \n{data}")
    return Command(
        update={
            "success_messages": data.get('success_messages'),
            "file_content": data.get('file_content'),
            "file_path": data.get('file_path'),
            "commit_message": data.get('commit_message'),
            "file_name": data.get('file_name'),
            "messages": result.get('messages'),
        },
        goto="supervisor",
    )


def create_commit_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_commit], prompt=create_commit_prompt)
    print(f"---------- COMMIT AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    retry_call = result['messages'][-1].tool_calls and not result['messages'][-1].content
    while not retry_call:
        result = agent.invoke(state)
        retry_call = result['messages'][-1].tool_calls and not result['messages'][-1].content
    return Command(
        update={"messages": result['messages']},
        goto="supervisor",
    )


def create_repo_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_github_repo], prompt=create_repo_prompt)
    print(f"----------REPO AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
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
    return Command(
        update={"messages": result['messages']},
        goto="supervisor",
    )

def supervisor_agent(state: SharedState) -> Command[Literal["create_repo", "create_readme_file", "create_commit", "generate_code", "__end__"]]:
    messages = [{"role": "system", "content": supervisor_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
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
}, config=RunnableConfig(recursion_limit=6)):
    print(f"----------STREAM------------\n")
