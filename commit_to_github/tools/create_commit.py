import os
from typing import Annotated, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from github import Github, Auth
from pydantic.v1 import BaseModel

load_dotenv()


class CreateCommitSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository to create"]
    file_path: Annotated[str, "Path of the file to be committed"] = "README.md"
    file_content: Annotated[str, "Content of the file to be committed"]
    commit_message: Annotated[Optional[str], "Message of the commit. Default is None"] = None
    branch_name: Annotated[Optional[str], "Name of the branch. Default is None"] = None


@tool
def create_commit(commit_data: CreateCommitSchema):
    """
    Tool to create a commit in a GitHub repository.
    """
    repo_name = commit_data.repo_name
    commit_message = commit_data.commit_message
    branch_name = commit_data.branch_name
    file_content = commit_data.file_content
    file_path = commit_data.file_path

    auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
    github_obj = Github(auth=auth)
    repo = github_obj.get_repo(repo_name)
    commit_obj = repo.create_file(path=file_path, content=file_content, message=commit_message, branch=branch_name)
    return {
        "commit_sha": commit_obj.get('commit').sha,
        "file_path": file_path,
        "file_content": file_content,
        "messages": f"Successfully created commit in {repo_name} at {branch_name} with file {file_path}"
    }
# commit = create_commit(schema=CreateCommitSchema(repo_name="kevit-hetvee-shah/test16", commit_message="test commit 3", branch_name="main", file_path="/home/kevit/PycharmProjects/assistants_agents/git_repo/add_initial_readme_commit/tools/README.md", file_content="TEST CONTENT"))
