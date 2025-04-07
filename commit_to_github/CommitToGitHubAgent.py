from git_repo.utils.Agent import Agent
from .tools.create_commit import create_commit

class CommitToGitHubAgent(Agent):
    def __init__(self):
        super().__init__(
            name="CommitToGitHubAgent",
            description="CommitToGitHubAgent is an agent that creates a commit to the GitHub repository.",
            instructions="./instructions.md",
            model_tpe="google_genai",
            tools=[create_commit],
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
