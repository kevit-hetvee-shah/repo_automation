import os

from github import Auth, Github
from langchain_core.tools import tool
from pydantic import BaseModel


class CreateCommitSchema(BaseModel):
    repo_name: str
    # organization_name: str
    commit_message: str
    branch_name: str
    file_content: str
    file_path: str


@tool
def create_commit(commit_data: CreateCommitSchema):
    """
    Tool to create a commit in a GitHub repository with given file path and content.
    """
    print(f"---------COMMIT DATA---------------\n{commit_data} ")
    try:

        commit_message = commit_data.commit_message
        branch_name = commit_data.branch_name
        file_content = commit_data.file_content
        repo_name = commit_data.repo_name
        file_path = commit_data.file_path
        auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
        github_obj = Github(auth=auth)

        # if "/" in commit_data.repo_name:
        #     repo_name = commit_data.repo_name
        # else:
        #     if commit_data.organization_name:
        #         repo_name = f"{commit_data.organization_name}/{commit_data.repo_name}"
        #     else:
        #         user = github_obj.get_user()
        #         repo_name = f"{user.login}/{commit_data.repo_name}"

        repo = github_obj.get_repo(repo_name)
        commit_obj = repo.create_file(path=file_path, content=file_content, message=commit_message, branch=branch_name)
        print(
            f"-------\ncommit_sha: f{commit_obj.get('commit').sha} \nfile_path: {file_path}\n file_content: {file_content}\n success_messages: Successfully created commit in {repo_name} at {branch_name} with file {file_path}\n----------------")

        return {
            "commit_sha": commit_obj.get('commit').sha,
            "file_path": file_path,
            "file_content": file_content,
            "success_messages": f"Successfully created commit in {repo_name} at {branch_name} with file {file_path}"
        }
    except Exception as e:
        print(f"Error in creating commit: {e}")
        breakpoint()
        return {
            "commit_sha": None,
            "file_path": None,
            "file_content": None,
            "success_messages": f"Error in creating commit: {e}"
        }
