import json

from github import UnknownObjectException
import os
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from github import Github, Auth
from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel

load_dotenv()


class DebugCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("Prompt sent to LLM:")
        for prompt in prompts:
            print(prompt)

    def on_llm_end(self, response, **kwargs):
        print("Response from LLM:")
        print(response)


# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ.get('GOOGLE_API_KEY'))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
    # callbacks=[DebugCallbackHandler()],
    verbose=True,
    # other params...
)


class SharedState(MessagesState):
    repo_name: Annotated[str, "Name of the repository."]
    extra_info: Annotated[str, "Additional information to be included in the README file."] = None
    file_path: Annotated[str, "Path of the file to be committed"]
    file_content: Annotated[str, "Content of the file to be committed"]
    commit_message: Annotated[Optional[str], "Message of the commit. Default is None"] = None
    branch_name: Annotated[Optional[str], "Name of the branch. Default is None"] = None
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created."
                        "Default is None")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True
    success_messages: Annotated[str, "Success messages from the tools."] = None


class CreateReadmeFileSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository."]
    extra_info: Annotated[str, "Additional information to be included in the README file."] = None


class CreateCommitSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository to create"]
    file_path: Annotated[str, "Path of the file to be committed"] = "README.md"
    file_content: Annotated[str, "Content of the file to be committed"]
    commit_message: Annotated[Optional[str], "Message of the commit. Default is None"] = None
    branch_name: Annotated[Optional[str], "Name of the branch. Default is None"] = None


class RepoCreationSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository to create"]
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
    print(readme_data, "README DATA")
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
    print(f"""{
    "success_messages": f"Successfully created the README file for the {repo_name} repository.",
        "file_content": content,
        "repo_name": repo_name
    }""")
    return {
        "success_messages": f"Successfully created the README file for the {repo_name} repository.",
        "file_content": content,
        "repo_name": repo_name
    }


@tool
def create_commit(commit_data: CreateCommitSchema):
    """
    Tool to create a commit in a GitHub repository.
    """
    print(commit_data, "COMMIT DATA")
    repo_name = commit_data.repo_name
    commit_message = commit_data.commit_message
    branch_name = commit_data.branch_name
    file_content = commit_data.file_content
    file_path = commit_data.file_path

    auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
    github_obj = Github(auth=auth)
    repo = github_obj.get_repo(repo_name)
    commit_obj = repo.create_file(path=file_path, content=file_content, message=commit_message, branch=branch_name)
    print(f"""{
    "commit_sha": commit_obj.get('commit').sha,
        "file_path": file_path,
        "file_content": file_content,
        "success_messages": f"Successfully created commit in {repo_name} at {branch_name} with file {file_path}"
    }""")
    return {
        "commit_sha": commit_obj.get('commit').sha,
        "file_path": file_path,
        "file_content": file_content,
        "success_messages": f"Successfully created commit in {repo_name} at {branch_name} with file {file_path}"
    }


@tool
def create_github_repo(repo_data: RepoCreationSchema):
    """
    Tool to create a GitHub repository.
    """
    print(repo_data, "REPO DATA")
    repo_name = repo_data.repo_name
    organization_name = repo_data.organization_name
    description = repo_data.description
    private = repo_data.private
    auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
    github_obj = Github(auth=auth)
    repos = len([i for i in github_obj.get_user().get_repos()])
    print(f"Currently {repos} exists.")

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
                                        description=description,
                                        private=private)
            repo = new_repo
    github_obj.close()
    print(f"""return {
    "success_messages": f"Successfully created the {repo.full_name} repository in github.",
        "repo_name": repo.full_name
    }""")
    return {
        "success_messages": f"Successfully created the {repo.full_name} repository in github.",
        "repo_name": repo.full_name
    }


def format_state(state: dict) -> str:
    return json.dumps(state, indent=2)


readme_agent_prompt = """# CreateReadmeContentAgent Instructions

You are an agent responsible for creating the README file content for the given repository.
Your responses should only address the creation of the content for README file. 
Do not include data from any external context. Operate as a standalone assistant focused solely on this task.

# Primary Instructions
1. You must use the create_readme_file tool to create the content of the README file.

## field names:
- repo_name: name of the repository and its required.
- extra_info: any extra information that is required to be included in the README file."""

create_commit_prompt = """# AddInitialREADMECommitAgent Instructions

You are an agent responsible for creating the README file content and creating a commit with the file in a repository.
Your responses should only address the creation of the content for README file and the commit process. 
Do not include data from any external context. Operate as a standalone assistant focused solely on these tasks.

# Primary Instructions
1. You must use the create_commit tool to create the commit.
2. Make sure the commit message reflects in the commit.
3. Do not create any other files or perform any other actions outside of creating the README file and committing it.

## field names:
- repo_name: name of the repository and its required.
- file_path: path where the README file will be created.
- file_content: content of the file.
- commit_message: message for the commit.
- branch_name: name of the branch where the commit will be made.
"""

create_repo_prompt = """# CreateRepoAgent Instructions

You are an agent responsible for creating a repository in the GitHub. You will be given details for creating the repository.
Using those details, create a repository.
Use create_repo_tool for this task

# Primary Instructions
1. You must use the create_repo_tool.
2. The field values will be provided to you. Consider them to create the repository.


## field names:
- repo_name: name of the repository and its required.
- description: description of the repository. Its optional.
- organization_name: name of the organization. If repository needs to be created in an organization, this field is required. Else repository will be created in the user account.
- private: whether the repository is private or not. Its optional. The default value is true."""

generate_code_prompt = """## GenerateCodeAgent Instructions

You are an agent responsible for generating the code. You will be given a prompt. For that prompt, you will generate the code. There should be no errors in the code.


# Primary Instructions
1. You must generate code in python language only.
2. The code must not have any errors.
3. The code must be well formatted.
4. The code must be well commented.
5. The code must contain at least 1 example. 


## field names:
- prompt: The prompt for which the code will be generated."""


def readme_agent(state: SharedState):
    agent = create_react_agent(llm, tools=[create_readme_file], prompt=readme_agent_prompt)
    agent.invoke(state)
    return Command(
        update={"messages": state['messages']},
        goto="create_commit",
    )


def create_commit_agent(state: SharedState):
    agent = create_react_agent(llm, tools=[create_commit], prompt=create_commit_prompt)
    agent.invoke(state)
    return Command(
        update={"messages": state['messages']},
        goto="generate_code",
    )


def create_repo_agent(state: SharedState):
    agent = create_react_agent(llm, tools=[create_github_repo], prompt=create_repo_prompt)
    agent.invoke(state)
    return Command(
        update={"messages": state['messages']},
        goto="create_readme_file",
    )


def generate_code_agent(state: SharedState):
    agent = create_react_agent(llm, tools=[], prompt=generate_code_prompt)
    agent.invoke(state)
    return Command(
        update={"messages": state['messages']},
        goto="create_commit",
    )


builder = StateGraph(SharedState)
builder.add_node("create_readme_file", readme_agent)
builder.add_node("create_commit", create_commit_agent)
builder.add_node("create_repo", create_repo_agent)
builder.add_node("generate_code", generate_code_agent)

builder.add_edge(START, "create_repo")

graph = builder.compile()
query = "Create a repo in the account, generate readme file and Generate code for fibonacci."
for q in graph.stream({
    "messages": [
        HumanMessage(role="user", content=query)
    ],
    "repo_name": "fibonacci_series",
    "extra_info": "This is a fibonacci series repo.",
    "file_path": "README.md",
    "file_content": "",
    "commit_message": "Initial commit",
    "branch_name": "main",
    "description": "This is a fibonacci series repo.",
    "organization_name": None,
    "private": True,
}):
    print(f"QQQQQL {q}")
    print("\n")
