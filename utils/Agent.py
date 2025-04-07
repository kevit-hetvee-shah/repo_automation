import json
import os
from typing import List, Dict, Any

from langchain.agents import create_react_agent
from dotenv import load_dotenv
import inspect
from langgraph_agents_dynamically.utils.state import SharedState
from langgraph_agents_dynamically.utils.model_type_mapping import MODEL_TYPES

load_dotenv()


class Agent:
    _shared_state: SharedState = None

    # @property
    # def assistant(self):
    #     if not hasattr(self, '_assistant') or self._assistant is None:
    #         raise Exception("Assistant is not initialized. Please run get_or_create_agent() first.")
    #     return self._assistant
    #
    # @assistant.setter
    # def assistant(self, value):
    #     self._assistant = value
    #
    # @property
    # def functions(self):
    #     return [tool for tool in self.tools]
    #
    # @property
    # def shared_state(self):
    #     return self._shared_state
    #
    # @shared_state.setter
    # def shared_state(self, value):
    #     self._shared_state = value
    #     for tool in self.tools:
    #         tool.shared_state = value
    #
    # def response_validator(self, message: str | list) -> str:
    #     """
    #     Validates the response from the agent. If the response is invalid, it must raise an exception with instructions
    #     for the caller agent on how to proceed.
    #
    #     Parameters:
    #         message (str): The response from the agent.
    #
    #     Returns:
    #         str: The validated response.
    #     """
    #     return message

    def __init__(
            self,
            id: str = None,
            name: str = None,
            description: str = None,
            instructions: str = None,
            model_tpe: str = None,
            tools: List = None,
            temperature: float = None,
            top_p: float = None,
            metadata: Dict[str, str] = None,
            model: str = "gemini-2.0-flash-lite",
            max_prompt_tokens: int = None,
            max_completion_tokens: int = None,
            truncation_strategy: dict = None,
            tools_folder: str = None

    ):
        """
        Initialize an Agent
        :param id: ID of the agent. Agent will be created if ID is not provided, else loaded. Defaults to None.
        :param name: Name of the agent. Defaults to the class name if not provided.
        :param description: Description of the agent
        :param instructions: Instructions for the agent. Can be file path or string.
        :param tools: List of tools that the agent can use
        :param temperature: Temperature for the agent
        :param top_p: Top p for the agent
        :param metadata: Metadata for the agent
        :param model: Model for the agent
        :param max_prompt_tokens: Max prompt tokens for the agent
        :param max_completion_tokens: Max completion tokens for the agent
        :param truncation_strategy: Truncation strategy for the agent

        tool_resources, response_format, tools_folder, files_folder, schemas_folder, api_headers, api_params,file_ids,
        validation_attempts, examples
        """
        self.id = id
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools = tools
        self.temperature = temperature
        self.top_p = top_p
        self.metadata = metadata
        self.model = model
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.truncation_strategy = truncation_strategy
        self.model_type = model_tpe

        self.settings_path = './settings.json'

        self._assistant: Any = None
        self._shared_instructions = None
        print("_______________")
        self._read_instructions()

    def get_or_create_agent(self):
        breakpoint()
        if self.name is None:
            self.name = self.__class__.__name__
        agent = self.retrieve_agent(self.name)
        breakpoint()
        if agent:
            return self
        else:
            if self.model_type == "google_genai":
                from langchain_google_genai import ChatGoogleGenerativeAI
                possible_models = MODEL_TYPES.get(self.model_type).get("chat_models")
                if self.model not in possible_models:
                    raise ValueError(
                        f"Model {self.model} is not supported for Google GenAI. Please choose one of the following models: {possible_models}")
                self.assistant = ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_completion_tokens,
                    max_retries=3,
                    google_api_key=os.environ.get('GOOGLE_API_KEY'),
                    metadata=self.metadata,
                )
                if self.tools:
                    self.assistant = create_react_agent(llm=self.assistant, tools=self.tools, prompt=self.instructions)

                assistant_data = [
                    {
                        "id": self.id,
                        "name": self.name,
                        "description": self.description,
                        "instructions": self.instructions,
                        "tools": self.tools,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "metadata": self.metadata,
                        "model": self.model,
                        "max_prompt_tokens": self.max_prompt_tokens,
                        "max_completion_tokens": self.max_completion_tokens,
                        "truncation_strategy": self.truncation_strategy,
                        "model_type": self.model_type,
                        "settings_path": self.settings_path,
                    }
                ]
                self.save_settings(assistant_data)

            else:
                print(f"Cant create agent for {self.model_type} model currently.")
                self.assistant = None
            breakpoint()
            return self.assistant

    def get_settings_path(self):
        return self.settings_path

    def read_settings(self):
        settings_path = self.get_settings_path()
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            return settings
        else:
            return []

    def get_class_folder_path(self):
        try:
            # First, try to use the __file__ attribute of the module
            return os.path.abspath(os.path.dirname(self.__module__.__file__))
        except (TypeError, OSError, AttributeError) as e:
            # If that fails, fall back to inspect
            try:
                class_file = inspect.getfile(self.__class__)
            except (TypeError, OSError, AttributeError) as e:
                return "./"
            return os.path.abspath(os.path.realpath(os.path.dirname(class_file)))

    def _read_instructions(self):
        class_instructions_path = os.path.normpath(os.path.join(self.get_class_folder_path(), self.instructions))
        if os.path.isfile(class_instructions_path):
            with open(class_instructions_path, 'r') as f:
                self.instructions = f.read()
        elif os.path.isfile(self.instructions):
            with open(self.instructions, 'r') as f:
                self.instructions = f.read()
        elif "./instructions.md" in self.instructions or "./instructions.txt" in self.instructions:
            raise Exception("Instructions file not found.")

    def save_settings(self, data):
        settings_path = self.get_settings_path()
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                updated_data = settings + data
                with open(settings_path, 'w') as f:
                    json.dump(updated_data, f)
            return settings
        else:
            return []

    def retrieve_agent(self, agent_name):
        settings = self.read_settings()
        if settings:
            agent = [agent for agent in settings if agent['name'] == agent_name]
            if agent:
                return agent[0]
        else:
            return None

    # def invoke(self, message):
    #     message = f"""
    #     You are a helpful assistant. Your task is to assist the user in any way possible. The user query is {message}.
    #     """
    #     if self.assistant:
    #         response = self.assistant.invoke(message)
    #         return response
    #     else:
    #         raise Exception("Assistant is not initialized. Please run get_or_create_agent() first.")

