from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import BaseModel
import os

from github import Auth, Github, UnknownObjectException

class RepoCreationSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository"]
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created."
                        "Default is None")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True


@tool
def create_github_repo(repo_data: RepoCreationSchema):
    """
    Tool to create a GitHub repository using given data.
    """
    print(f"---------------REPO DATA--------------\n{repo_data} ")
    try:
        repo_name = repo_data.repo_name
        organization_name = repo_data.organization_name
        description = repo_data.description
        private = repo_data.private
        auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
        github_obj = Github(auth=auth)
        if organization_name:
            organization = github_obj.get_organization(organization_name)
            try:
                existing_repo = organization.get_repo(repo_name)
                repo = existing_repo
            except UnknownObjectException as e:
                new_repo = organization.create_repo(
                    name=repo_name,
                    allow_rebase_merge=True,
                    description=description,
                    private=private,
                )
                repo = new_repo
        else:
            user = github_obj.get_user()
            try:
                existing_repo = user.get_repo(repo_name)
                repo = existing_repo
            except UnknownObjectException as e:
                new_repo = user.create_repo(name=repo_name,
                                            description=description if description else "Repository created by langgraph",
                                            private=private)
                repo = new_repo
        github_obj.close()

        print(
            f"--------\nMessage: Successfully created the {repo.full_name} repository in github. \nrepo_name: {repo.full_name}\n-------------")
        return {
            "success_messages": f"Successfully created the {repo.full_name} repository in github.",
            "repo_name": repo.full_name,

        }
    except Exception as e:
        print(f"Error in creating repository: {e}")
        breakpoint()
        return {
            "success_messages": f"Error in creating repository: {e}",
            "repo_name": None,
        }
