from git_repo.utils.Agent import Agent
from .tools.create_readme_file import create_readme_file

class CreateReadmeContent(Agent):
    def __init__(self):
        super().__init__(
            name="CreateReadmeContentAgent",
            description="CreateReadmeContentAgent is an agent that adds a README file to the repository.",
            instructions="./instructions.md",
            model_tpe="google_genai",
            tools=[create_readme_file],
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
