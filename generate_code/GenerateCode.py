from git_repo.utils.Agent import Agent


class GenerateCode(Agent):
    def __init__(self):
        super().__init__(
            name="GenerateCode",
            description="GenerateCode is an agent that generates code for a given prompt. Make sure that the code is in python language.",
            instructions="./instructions.md",
            model_tpe="google_genai",
            tools=[],
            temperature=0.5,
            top_p=0.5,
            metadata={},
            model="gemini-2.0-flash-lite",
        )