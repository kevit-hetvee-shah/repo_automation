import json

from github import UnknownObjectException
import os
from typing import Optional, List, TypedDict

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


def create_llm(
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = 2,
        max_retries: Optional[int] = 2,
        timeout: Optional[int] = None,
        verbose: bool = True,
):
    """
    Create a Google Generative AI LLM instance.

        :param: model (str): The model to use (default: "gemini-2.0-flash-lite").
        :param: temperature (float): Sampling temperature (default: 0).
        :param: max_tokens (int, optional): Maximum number of tokens.
        :param: top_p (float, optional): Top-p sampling parameter.
        :param: top_k (int, optional): Top-k sampling parameter.
        :param: max_retries (int, optional): Maximum retry attempts.
        :param: timeout (int, optional): Maximum wait until timeout.
        :param: verbose (bool): Whether to print verbose output (default: True).
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_retries=max_retries,
        timeout=timeout,
        verbose=verbose,
        max_tokens=max_tokens,
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
    )


class State(TypedDict):
    """Shared state for the agent."""
    messages: List[dict]


def generate_agent_prompt(prompt_instructions: str):
    """Generate the agent prompt."""
    return f"""You are a helpful AI assistant, collaborating with other assistants.
         Use the provided tools to progress towards answering the question.
         If you are unable to fully answer, that's OK, another assistant with different tools 
         will help where you left off. Execute what you can to make progress.
         If you or any of the other assistants have the final answer or deliverable,
         prefix your response with FINAL ANSWER so the team knows to stop.
        \n`{prompt_instructions}`"""


def create_agent(
        name: str,
        prompt: str,
        model,
        response_format,
        tools: Optional[List] = None,

):
    return create_react_agent(name=name, model=model, tools=tools, prompt=prompt, response_format=response_format)


llm = create_llm()


class SharedState(State):
    """Shared state for the agent."""
    next: str
