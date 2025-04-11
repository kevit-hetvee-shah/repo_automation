import os

from general.agents.utils.agent import Agent

class SupervisorAgent(Agent):

    def __init__(self):
        agent_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "instructions.md")
        super().__init__(agent_name = "SupervisorAgent", agent_prompt = agent_path, tools = [], is_supervisor=True)