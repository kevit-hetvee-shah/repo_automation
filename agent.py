import json

from github import UnknownObjectException
import os
from typing import Optional, List, TypedDict, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
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


class Agent:

    def __init__(self, agent_name, agent_prompt, model="gemini-2.0-flash-lite", temperature=None, max_tokens=None,
                 top_p=None, top_k=None, max_retries=None, timeout=None, verbose=None, response_format=None,
                 tools: Optional[List] = None):
        """
        :param: agent_name (str): .
        :param: agent_prompt (str): Prompt for the agent.
        :param: model (str): The model to use (default: "gemini-2.0-flash-lite").
        :param: temperature (float): Sampling temperature (default: 0).
        :param: max_tokens (int, optional): Maximum number of tokens.
        :param: top_p (float, optional): Top-p sampling parameter.
        :param: top_k (int, optional): Top-k sampling parameter.
        :param: max_retries (int, optional): Maximum retry attempts.
        :param: timeout (int, optional): Maximum wait until timeout.
        :param: verbose (bool): Whether to print verbose output (default: True).
        :param: response_format (str): The response format.
        :param: tools (List, optional): List of tools to bind to llm.
        """
        self.name = agent_name
        self.response_format = response_format
        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.max_retries = max_retries or 5
        self.timeout = timeout
        self.verbose = verbose
        if os.path.exists(agent_prompt):
            self.prompt = open(agent_prompt, "r").read()
        else:
            self.prompt = agent_prompt

    def create_llm(self):
        """
        Create a Google Generative AI LLM instance.
        """
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_retries=self.max_retries,
            timeout=self.timeout,
            verbose=self.verbose,
            max_tokens=self.max_tokens,
            google_api_key=os.environ.get('GOOGLE_API_KEY'),
        )

    def create_agent(self):
        """
        Create an agent using the create_react_agent function.
        """
        return create_react_agent(
            name=self.name,
            model=self.model,
            tools=self.tools,
            prompt=self.prompt,
            response_format=self.response_format
        )

    def create_prompt(self):
        """Generate the agent prompt."""
        return f"""You are a helpful AI assistant, collaborating with other assistants.
             Use the provided tools to progress towards answering the question.
             If you are unable to fully answer, that's OK, another assistant with different tools 
             will help where you left off. Execute what you can to make progress.
             If you or any of the other assistants have the final answer or deliverable,
             prefix your response with FINAL ANSWER so the team knows to stop.
            \n`{self.prompt}`"""


class SharedState(TypedDict):
    """Shared state for the agent."""
    next: str
    messages: list[dict]


def create_repo_agent(state: SharedState) -> Command[Literal[str]]:
    repo_agent = Agent(agent_name="create_repo_agent", agent_prompt="").create_agent()
    result = repo_agent.invoke(state["messages"])
    return Command(
        update={
            "messages": result.get('messages'),
        },
        goto="supervisor",
    )
