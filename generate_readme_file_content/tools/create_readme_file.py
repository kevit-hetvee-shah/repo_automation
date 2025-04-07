from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel


class CreateReadmeFileSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository."]
    extra_info: Annotated[str, "Additional information to be included in the README file."] = None

@tool
def create_readme_file(readme_data: CreateReadmeFileSchema):
    """
    Tool to create the content of README file.
    """
    repo_name = readme_data.repo_name
    extra_info = readme_data.extra_info
    content = f"""
    # {repo_name}\nThis is the README file for the {repo_name} repository.

        ## Prerequisite: \n- Python 3.8+

        ## Installation:
        \n```bash\n
        pip3 install -r requirements.txt\n
        ```

        ## Run code: \n
        ```bash\n
        python3 main.py\n
        ```
        """
    if extra_info:
        content += f"\n##Details:\n{extra_info}\n"
    return {
        "messages": f"Successfully created the README file for the {repo_name} repository.",
        "file_content": content,
        "repo_name": repo_name
    }

# file = create_readme_file(repo_name="kevit-hetvee-shah/test16")
