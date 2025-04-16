# Supervisor-Based Agent System for GitHub Automation

This project implements a supervisor-based agent system designed to automate GitHub-related tasks. It leverages LangChain, Streamlit, and other libraries to create a modular, state-driven workflow for managing repositories, commits, and other GitHub operations.

## Key Features

- **Modular Agents**: Specialized agents for tasks such as creating repositories, commits, and README files.
- **Supervisor Agent**: Orchestrates task execution by delegating to other agents.
- **State Management**: Integration with LangChain for managing state and tool invocation.
- **Streamlit UI**: Provides an interactive user interface for input and workflow execution.

### Agents Overview
- create_repo: Creates a new GitHub repository.
- create_readme_file: Generates the content for a README file.
- create_commit: Creates a commit in a specified repository.
- generate_code: Generates Python code based on a given prompt.
- list_repo_branches: Lists all branches of a specified repository.
- general_assistant: Handles general-purpose tasks and queries.

### Workflow
The system uses a supervisor agent to analyze user input and delegate tasks to the appropriate agents. The workflow is managed using a state graph, ensuring modularity and flexibility.


### Checkpointing and Memory
- In-Memory Store: Used for temporary storage of state and checkpoints.
- SQLite Checkpointing: Provides persistent storage for workflow checkpoints.

#### Example Query
- Create a Repository: "Create a private repository named my-repo with a description 'My new project'."
- Generate a README: "Generate a README file for the repository my-repo."

## Prerequisites

- Python 3.8 or higher
- A valid GitHub access token
- Google API key for LLM integration

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
   
2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install requirements
    ```bash 
    pip install -r requirements.txt
    ```

4. Create a .env file in the root directory and add the environment variables defined in example.env:

### Run the Streamlit application:

```bash 
   streamlit run supervisor_based_1.py
```

Enter your query in the input field and click the "Run" button to execute the workflow.


