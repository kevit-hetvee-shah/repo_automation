import os

from general.agents.utils.agent import Agent
from .tools.create_repo import create_github_repo

class CreateGitHubRepoAgent(Agent):

    def __init__(self):
        agent_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "instructions.md")
        super().__init__(agent_name = "CreateGitHubRepoAgent", agent_prompt = agent_path, tools = [create_github_repo])