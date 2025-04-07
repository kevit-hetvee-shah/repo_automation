from git_repo.utils.Agent import Agent
from .tools.create_repo_tool import create_github_repo

class CreateRepoAgent(Agent):
    def __init__(self):
        super().__init__(
            name="CreateRepoAgent",
            description="CreateRepoAgent is a agent that creates a repository in the github with given details",
            instructions="./instructions.md",
            model_tpe="google_genai",
            tools=[create_github_repo],
            temperature=0.5,
            top_p=0.5,
            metadata={},
            model="gemini-2.0-flash-lite",
            # max_prompt_tokens=1000,
            # max_completion_tokens=1000,
            # truncation_strategy={
            #     "type": "truncate_by_tokens",
            #     "max_tokens": 1000
            # }
        )
