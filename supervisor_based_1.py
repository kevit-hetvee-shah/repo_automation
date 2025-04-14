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
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
load_dotenv()

members = ["create_repo", "create_readme_file", "create_commit", "generate_code", "list_repo_branches",
           "general_assistant"]

options = members + ["FINISH"]


class Router(BaseModel):
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


class SharedState(MessagesState):
    # for all the tools: success_messages
    success_messages: Annotated[str, "Success messages from the tools."] = None
    # repo creation: repo_name, description, organization_name, private
    repo_name: Annotated[str, "Name of the repository."] = None
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created.")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True
    # readme content: repo_name, extra_readme_info
    extra_readme_info: Annotated[str, "Additional information to be included in the README file."] = None
    # commit: repo_name, file_path, file_content, commit_message, branch_name
    branch_name: Annotated[Optional[str], "Name of the branch. Default is main"] = None
    file_name: Annotated[str, "Name of the file to be committed"] = None
    file_path: Annotated[str, "Path of the file to be committed"] = None
    file_content: Annotated[str, "Content of the file to be committed"] = None
    commit_message: Annotated[Optional[str], "Message of the commit. Default is None"] = None


@tool
def create_readme_file(repo_name: str, extra_info: str) -> str:
    """
    Tool to create the content of README file.
    """
    print(f"-----------README DATA-----------\nREPO NAME: {repo_name}, EXTRA INFO: {extra_info}")
    try:
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
            "repo_name": repo_name,
            "extra_info": extra_info,
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
            "repo_name": repo_name,
            "extra_info": extra_info,
        }


@tool
def create_commit(repo_name: str, file_path: str, file_content: str,
                  commit_message: str, branch_name: str = None) -> str:
    """
    Tool to create a commit in a GitHub repository with given file path and content.
    """
    print(f"---------COMMIT DATA---------------\nREPO NAME: {repo_name}, FILE PATH: {file_path}, "
          f"FILE CONTENT: {file_content}, COMMIT MESSAGE: {commit_message}, BRANCH NAME: {branch_name}")
    try:
        auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
        github_obj = Github(auth=auth)
        repo = github_obj.get_repo(repo_name)
        commit_obj = repo.create_file(path=file_path, content=file_content, message=commit_message, branch=branch_name)
        print(
            f"-------\ncommit_sha: f{commit_obj.get('commit').sha} \nfile_path: {file_path}\n file_content: {file_content}\n success_messages: Successfully created commit in {repo_name} at {branch_name} with file {file_path}\n----------------")

        return {
            "success_messages": f"Successfully created commit in {repo_name} at {branch_name} with file {file_path}",
            "file_path": file_path,
            "file_content": file_content,
            "commit_message": commit_message,
            "branch_name": branch_name,
            "repo_name": repo_name,
        }
    except Exception as e:
        print(f"Error in creating commit: {e}")
        breakpoint()
        return {
            "success_messages": f"Error in creating commit: {e}",
            "file_path": file_path,
            "file_content": file_content,
            "commit_message": commit_message,
            "branch_name": branch_name,
            "repo_name": repo_name,
        }


@tool
def create_github_repo(repo_name: str, description: str, organization_name: str, private: bool) -> str:
    """
    Tool to create a GitHub repository using given data.
    """
    print(f"---------------REPO DATA--------------\nREPO NAME  {repo_name}, DESCRIPTION: {description}, "
          f"ORGANIZATION NAME: {organization_name}, PRIVATE: {private}")
    try:
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
            "description": description,
            "organization_name": organization_name,
            "private": private,
        }

    except Exception as e:
        print(f"Error in creating repository: {e}")
        breakpoint()
        return {
            "success_messages": f"Error in creating repository: {e}",
            "repo_name": None,
            "description": description,
            "organization_name": organization_name,
            "private": private,
        }


@tool
def list_repo_branches_tool():
    """
    Tool to list branches of a GitHub repository.
    """
    human_response = input("Enter the repository name: ")
    print(f"---------REPO DATA--------------\nREPO NAME: {human_response}")
    auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
    github_obj = Github(auth=auth)
    repo = github_obj.get_repo(human_response)
    branches = [i.name for i in list(repo.get_branches())]
    print(f"Branches: {branches}")
    return {"branches": branches, "human_input": human_response}


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
- If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
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
- If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."

"""

create_repo_prompt = """# CreateRepoAgent Instructions

You are an agent responsible for creating a repository in the GitHub. 

# Primary Instructions
1. You must use the create_repo_tool to create the repository.

## field names:
- repo_name: name of the repository.
- description: description of the repository.
- organization_name: name of the organization. If not provided, it will be created in the user's account. Consider as None if not provided.
- private: whether the repository is private or not.

# NOTE:
- You should not ask any questions to the user. Just simple create the repository with the given instructions.
- Just use tools and complete the task that can be completed by you and return the response. Other tools will do the rest of the work.
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.
- If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
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
- If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
"""

supervisor_prompt = f"""# Supervisor Instructions


# Primary Instructions
- You are an agent responsible for supervising the other agents.
- Your role is understand the user input and decide which agent to use.
- You have these agents with you {members}

# Agents Description:
- create_repo: This agent is responsible for creating a repository in the GitHub.
- create_readme_file: This agent is responsible for creating the README file content for the given repository.
- create_commit: This agent is responsible for creating a commit in the repository.
- generate_code: This agent is responsible for generating the code in python language.
- list_repo_branches: This agent is responsible for listing the branches of the given repository.
- general_assistant: This agent is responsible for general purpose tasks.

# NOTE:
- You must return the agent name in the response as it will be used by the agent to complete the task.
- If the agent response contains "FINAL ANSWER", then it means that the task is complete.
- When all tasks asked by user are complete, include "FINISH" as the final step.

# IMPORTANT
- If the agent's response contains "FINAL ANSWER" at beginning, middle or end, then it means that the agent task is complete. If any of the other agent can satisfy the user's query, call other agent else END the conversation.
- Analyze the response from agent including AIMessage and ToolMessage and decide what step to take next. 1. Call Agent or 2. End the conversation.
- If there's a need to call multiple agents, call them in order and define the sequence of agents to call.
"""

general_agent_prompt = """
You are an agent responsible for providing the general information.
Your task is provide the answer to the user's query.

# Primary Instructions
- You should not ask any questions to the user. Just simply provide the answer to the user's query.
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.
- If you are unable to fully answer, that's OK, another assistant with different tools "

# NOTE:
- If you don't know te answer, simply say you don't know. Don't try to make up an answer.

"""

list_repository_branches_prompt = """
    You are agent responsible for viewing all the branches of a repository.
    Do not include data from any external context. Operate as a standalone assistant focused solely on this task.

    
# Primary Instructions
- You must use the list_repository_branches_tool to list all the branches of a repository.
- You should not ask any questions to the user. Just simply call the tool

# NOTE:
- Just use tools and complete the task that can be completed by you and return the response. 
- If your job is done, prefix your response with "FINAL ANSWER" so the team knows to stop.

# IMPORTANT
- If your job is done, always prefix your response with "FINAL ANSWER" i.e add "FINAL ANSWER" to the beginning of the response.
- If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
"""


def readme_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_readme_file], prompt=readme_agent_prompt)
    print(f"---------- README AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            print(f"----------DATA------------: \n{data}")
            updated_state = {}
            for key, value in data.items():
                updated_state[key] = value
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
        except Exception as error:
            print(f"Error while parsing JSON: {error}")
            breakpoint()
            updated_state = {}
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
    else:
        print(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO tool call >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return Command(
            update={"messages": [HumanMessage(content=result['messages'][-1].content, name="generate_code")]},
            goto="supervisor",
        )


def create_commit_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_commit], prompt=create_commit_prompt)
    print(f"---------- COMMIT AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            print(f"----------DATA------------: \n{data}")
            updated_state = {}
            for key, value in data.items():
                updated_state[key] = value
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
        except Exception as error:
            print(f"Error while parsing JSON: {error}")
            breakpoint()
            updated_state = {}
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
    else:
        print(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO tool call >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return Command(
            update={"messages": [HumanMessage(content=result['messages'][-1].content, name="generate_code")]},
            goto="supervisor",
        )


def create_repo_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[create_github_repo], prompt=create_repo_prompt)
    print(f"----------REPO AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            print(f"----------DATA------------: \n{data}")
            updated_state = {}
            for key, value in data.items():
                updated_state[key] = value
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
        except Exception as error:
            print(f"Error while parsing JSON: {error}")
            breakpoint()
            updated_state = {}
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
    else:
        print(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO tool call >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return Command(
            update={
                "messages": [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            },
            goto="supervisor",
        )


def generate_code_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[], prompt=generate_code_prompt)
    print(f"----------GENERATE CODE STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            print(f"----------DATA------------: \n{data}")
            updated_state = {}
            for key, value in data.items():
                updated_state[key] = value
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
        except Exception as error:
            print(f"Error while parsing JSON: {error}")
            breakpoint()
            updated_state = {}
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
    else:
        print(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO tool call >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return Command(
            update={
                "messages": [HumanMessage(content=result['messages'][-1].content, name="generate_code")],
            },
            goto="supervisor",
        )


def list_repository_branches_agent(state: SharedState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[list_repo_branches_tool],
                               prompt=list_repository_branches_prompt)
    print(f"----------LIST REPO BRANCHES STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            print(f"----------DATA------------: \n{data}")
            updated_state = {}
            for key, value in data.items():
                updated_state[key] = value
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
        except Exception as error:
            print(f"Error while parsing JSON: {error}")
            breakpoint()
            updated_state = {}
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
    else:
        print(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO tool call >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return Command(
            update={
                "messages": [HumanMessage(content=result['messages'][-1].content, name="list_repo_branches")],
            },
            goto="supervisor",
        )


def general_agent(state: SharedState) -> Command[
    Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[], prompt=general_agent_prompt)
    print(f"---------- README AGENT STATE------------: \n{state}")
    result = agent.invoke(state)
    print(f"----------RESULT------------: \n{result}")
    tool_messages = [i for i in result['messages'] if isinstance(i, ToolMessage)]
    if tool_messages:
        tool_message = tool_messages[-1]
        try:
            data = json.loads(tool_message.content)
            print(f"----------DATA------------: \n{data}")
            updated_state = {}
            for key, value in data.items():
                updated_state[key] = value
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
        except Exception as error:
            print(f"Error while parsing JSON: {error}")
            breakpoint()
            updated_state = {}
            updated_state['messages'] = [HumanMessage(content=result['messages'][-1].content, name="generate_code")]
            return Command(
                update=updated_state,
                goto="supervisor",
            )
    else:
        print(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO tool call >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return Command(
            update={"messages": [HumanMessage(content=result['messages'][-1].content, name="generate_code")]},
            goto="supervisor",
        )


def supervisor_agent(state: SharedState) -> Command[
    Literal["create_repo", "create_readme_file", "create_commit", "generate_code", "__end__"]]:
    messages = [{"role": "system", "content": supervisor_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    print(
        f"-----------------------------------------------------------------SUPERVISOR RESPONSE----------------------------------------------------: \n{response}")
    goto = response.next
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
builder.add_node("list_repo_branches", list_repository_branches_agent)
builder.add_node("general_assistant", general_agent)

builder.add_edge(START, "supervisor")
import streamlit as st

st.header("Github Automation")
graph = builder.compile(checkpointer=memory)
query = st.text_input("Enter the query here: ")
if st.button("Run"):
    # Create a placeholder to display output; you can use st.empty() to update as you go.
    output_placeholder = st.empty()
    process_placeholder = st.empty()

    process_output = ""
    final_output = ""
    steps = []
    text_steps = []
    # Optional: use a spinner to indicate processing
    with st.spinner("Running the workflow..."):
        for output in graph.stream({
            "messages": [
                HumanMessage(role="user", content=query)
            ],
            "branch_name": "main",
            "success_messages": None,
            "repo_name": None,
            "description": None,
            "private": False,
            "extra_readme_info": None,
            "file_name": None,
            "file_content": None,
            "commit_message": None,
            "file_path": "",
            "organization_name": None
        }, config=RunnableConfig(recursion_limit=16, configurable={"thread_id": "1"})):
            steps.append(output)

            if 'supervisor' not in output.keys():
                for dict_values_item in output:
                    if dict_values_item in members:
                        data = output[dict_values_item]

                        messages = data.get("messages", [])
                        for msg in messages:
                            text_steps.append({dict_values_item: msg.content})
                output_placeholder.text_area("Output:", text_steps, height=300)
