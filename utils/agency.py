import inspect
import os
from typing import TypedDict, Annotated, List, Literal


class Agency:
    def __init__(self, agents: List, shared_instructions: str, temperature: float = 0.3,
                 top_p: float = 1.0,
                 max_prompt_tokens: int = None,
                 max_completion_tokens: int = None,
                 truncation_strategy: dict = None, settings_path: str = "./settings.json", ):
        self.agents = agents
        self.temperature = temperature
        self.top_p = top_p
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.truncation_strategy = truncation_strategy
        self.settings_path = settings_path

        if os.path.isfile(os.path.join(self._get_class_folder_path(), shared_instructions)):
            self._read_instructions(os.path.join(self._get_class_folder_path(), shared_instructions))
        elif os.path.isfile(shared_instructions):
            self._read_instructions(shared_instructions)
        else:
            self.shared_instructions = shared_instructions

        self._init_agents()

    def _read_instructions(self, path):
        """
        Reads shared instructions from a specified file and stores them in the agency.

        Parameters:
            path (str): The file path from which to read the shared instructions.

        This method opens the file located at the given path, reads its contents, and stores these contents in the 'shared_instructions' attribute of the agency. This is used to provide common guidelines or instructions to all agents within the agency.
        """
        path = path
        with open(path, 'r') as f:
            self.shared_instructions = f.read()

    def _get_class_folder_path(self):
        """
        Retrieves the absolute path of the directory containing the class file.

        Returns:
            str: The absolute path of the directory where the class file is located.
        """
        return os.path.abspath(os.path.dirname(inspect.getfile(self.__class__)))

    # def invoke(self, message):
    #     message = f"""
    #     You are a helpful assistant. Your task is to assist the user in any way possible. The user query is {message}.
    #     """
    #
    #     response = self.assistant.invoke(message)
    #     return response

    def _init_agents(self):
        """
        Initializes all agents in the agency with unique IDs, shared instructions, and OpenAI models.

        This method iterates through each agent in the agency, assigns a unique ID, adds shared instructions, and initializes the OpenAI models for each agent.

        There are no input parameters.

        There are no output parameters as this method is used for internal initialization purposes within the Agency class.
        """

        for agent in self.agents:

            agent.add_shared_instructions(self.shared_instructions)
            agent.settings_path = self.settings_path

            if self.shared_files:
                if isinstance(self.shared_files, str):
                    self.shared_files = [self.shared_files]

                if isinstance(agent.files_folder, str):
                    agent.files_folder = [agent.files_folder]
                    agent.files_folder += self.shared_files
                elif isinstance(agent.files_folder, list):
                    agent.files_folder += self.shared_files

            if self.temperature is not None and agent.temperature is None:
                agent.temperature = self.temperature
            if self.top_p and agent.top_p is None:
                agent.top_p = self.top_p
            if self.max_prompt_tokens is not None and agent.max_prompt_tokens is None:
                agent.max_prompt_tokens = self.max_prompt_tokens
            if self.max_completion_tokens is not None and agent.max_completion_tokens is None:
                agent.max_completion_tokens = self.max_completion_tokens
            if self.truncation_strategy is not None and agent.truncation_strategy is None:
                agent.truncation_strategy = self.truncation_strategy

            agent.get_or_create_agent()
