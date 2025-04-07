import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from github import Github, Auth, UnknownObjectException
from pydantic.v1 import BaseModel

load_dotenv()


class RepoCreationSchema(BaseModel):
    repo_name: Annotated[str, "Name of the repository to create"]
    description: Annotated[Optional[str], "Description of the repository. Default is None"] = None
    organization_name: Annotated[
        Optional[str], ("Name of the organization of the github in which repository needs to be created."
                        "Default is None")] = None
    private: Annotated[bool, "Whether the repository should be private or not. Default is True"] = True


@tool
def create_github_repo(repo_data: RepoCreationSchema):
    """
    Tool to create a GitHub repository.
    """
    repo_name = repo_data.repo_name
    organization_name = repo_data.organization_name
    description = repo_data.description
    private = repo_data.private
    auth = Auth.Token(os.environ.get("GITHUB_ACCESS_TOKEN"))
    github_obj = Github(auth=auth)
    repos = len([i for i in github_obj.get_user().get_repos()])
    print(f"Currently {repos} exists.")

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
                                        description=description,
                                        private=private)
            repo = new_repo
    github_obj.close()
    return {
        "messages": f"Successfully created the {repo.full_name} repository in github.",
        "repo_name": repo.full_name
    }

# repo = create_github_repo(schema = RepoCreationSchema(repo_name="test15", description="Test repo", organization_name=None, private=True))
