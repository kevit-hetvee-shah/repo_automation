import os

from general.agents.utils.agent import Agent
from .tools.create_readme_content import create_readme_file

class CreateReadmeContentAgent(Agent):

    def __init__(self):
        agent_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "instructions.md")
        super().__init__(agent_name = "CreateReadmeContentAgent", agent_prompt = agent_path, tools = [create_readme_file])