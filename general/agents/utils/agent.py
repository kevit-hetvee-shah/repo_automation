import json

from github import UnknownObjectException
import os
from typing import Optional, List, TypedDict, Literal

from langchain_community.callbacks import OpenAICallbackHandler
from langchain_openai import OpenAI, ChatOpenAI

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
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

callback_handler = OpenAICallbackHandler()

class DebugCallbackHandler(BaseCallbackHandler):

    def on_llm_end(self, response, **kwargs):
        print("-----------------Response from LLM:-----------------")
        print(f"RESPONSE: {response}")
        print("******************************************************")

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


class Agent:

    def __init__(self, agent_name, agent_prompt, is_supervisor=False, model=None, temperature=0,
                 max_tokens=2048, is_google=False,
                 top_p=0, top_k=None, max_retries=None, timeout=None, verbose=None, response_format=None,
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
        openai_model = "gpt-4o-mini"
        google_model = "gemini-2.0-flash-lite"
        self.name = agent_name
        self.response_format = response_format
        self.tools = tools
        self.model = google_model if is_google else openai_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.max_retries = max_retries or 5
        self.timeout = timeout
        self.verbose = verbose
        self.llm = self.create_google_llm() if is_google else self.create_openai_llm()
        if is_supervisor:
            if os.path.exists(agent_prompt):
                self.prompt = open(agent_prompt, "r").read()
            else:
                self.prompt = agent_prompt
        else:
            if os.path.exists(agent_prompt):
                self.prompt = SystemMessage(content=self.create_prompt(prompt=open(agent_prompt, "r").read()))
            else:
                self.prompt = SystemMessage(content=self.create_prompt(prompt=agent_prompt))

    def create_google_llm(self):
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
            callbacks=[callback_handler]
            # callback_manager=callback_manager,
        )

    def create_openai_llm(self):
        """
        Create a Google Generative AI LLM instance.
        """
        return ChatOpenAI(
            model_name=self.model,
            callbacks=[callback_handler],
            # temperature=self.temperature,
            # callbacks=[callback_handler],
            # top_p=self.top_p,
            # max_retries=self.max_retries,
            # request_timeout=self.timeout,
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            # max_tokens=self.max_tokens,
            callback_manager=callback_manager,
            # verbose=self.verbose,
            # model=self.model,
            # top_k=self.top_k,
            # timeout=self.timeout,
            # google_api_key=os.environ.get('GOOGLE_API_KEY'),
        )

    def create_agent(self):
        """
        Create an agent using the create_react_agent function.
        """
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.prompt,
            response_format=self.response_format
        )

    def create_prompt(self, prompt):
        """Generate the agent prompt."""
        return f"""# General Instructions: 
        
         You are a helpful AI assistant, collaborating with other assistants.
         Use the provided tools to progress towards answering the question.
         If you are unable to fully answer, that's OK, another assistant with different tools 
         will help where you left off. Execute what you can to make progress.
         If you or any of the other assistants have the final answer or deliverable,
         prefix your response with FINAL ANSWER so the team knows to stop.
        \n`{prompt}`"""
