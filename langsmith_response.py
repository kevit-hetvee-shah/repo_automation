from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

abc = [
    HumanMessage(
    content="\n    Create a repo in the named fibonacci_series. \n    The repo should contain a README.md file with the content \n    'This is a fibonacci series repo.' and the\n     file should be created in the main branch. \n     The commit message should be 'Initial commit'. \n     The repo should be private with no organization.",
    additional_kwargs={}, response_metadata={}, id='bfb2d4fb-5260-4208-a99b-e25e8d9c5b13', role='user'),


    AIMessage(content='',
              additional_kwargs={
                  'function_call': {'name': 'create_github_repo', 'arguments': '{"repo_data": "unknown"}'}},
              response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP',
                                 'safety_ratings': []}, id='run-fc0af5a7-edbf-4da9-9999-c0d8f6df34c3-0', tool_calls=[
            {'name': 'create_github_repo', 'args': {'repo_data': 'unknown'},
             'id': 'b43b0cd3-6bec-47ee-975b-5879337fb254',
             'type': 'tool_call'}], usage_metadata={'input_tokens': 339, 'output_tokens': 9, 'total_tokens': 348,
                                                    'input_token_details': {'cache_read': 0}}),


    ToolMessage(
        content="Error: 1 validation error for create_github_repo\nrepo_data\n  Input should be a valid dictionary or instance of RepoCreationSchema [type=model_type, input_value='unknown', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.10/v/model_type\n Please fix your mistakes.",
        name='create_github_repo', id='17c609d1-3740-4f7c-8267-090a49c9afda',
        tool_call_id='b43b0cd3-6bec-47ee-975b-5879337fb254', status='error'),


    AIMessage(content='I will create a private repository named fibonacci_series.', additional_kwargs={
        'function_call': {'name': 'create_github_repo',
                          'arguments': '{"repo_data": {"description": "This is a fibonacci series repo.", "repo_name": "fibonacci_series", "private": true}}'}},
              response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP',
                                 'safety_ratings': []}, id='run-d06bde5f-a6ee-404a-945f-fdedbfef7c47-0', tool_calls=[
            {'name': 'create_github_repo', 'args': {
                'repo_data': {'description': 'This is a fibonacci series repo.', 'repo_name': 'fibonacci_series',
                              'private': True}}, 'id': 'a5105394-9019-4d4d-9b96-986cd10f4ac0', 'type': 'tool_call'}],
              usage_metadata={'input_tokens': 435, 'output_tokens': 35, 'total_tokens': 470,
                              'input_token_details': {'cache_read': 0}}),


    ToolMessage(
        content='{"success_messages": "Successfully created the kevit-hetvee-shah/fibonacci_series repository in github.", "repo_name": "kevit-hetvee-shah/fibonacci_series"}',
        name='create_github_repo', id='f1dc8e48-5476-4c4b-9861-5003425a882a',
        tool_call_id='a5105394-9019-4d4d-9b96-986cd10f4ac0'),


    AIMessage(content='I have successfully created the repository named fibonacci_series.', additional_kwargs={},
              response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP',
                                 'safety_ratings': []}, id='run-48bfc6a2-4e1a-40bf-a302-fd6b60d1030e-0',
              usage_metadata={'input_tokens': 498, 'output_tokens': 12, 'total_tokens': 510,
                              'input_token_details': {'cache_read': 0}}),


    AIMessage(content=' Now I will create the README.md file.', additional_kwargs={
        'function_call': {'name': 'create_readme_file',
                          'arguments': '{"readme_data": {"repo_name": "fibonacci_series", "extra_info": "This is a fibonacci series repo."}}'}},
              response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP',
                                 'safety_ratings': []}, id='run-687fc100-098d-44d6-818b-71c4271ab888-0', tool_calls=[
            {'name': 'create_readme_file',
             'args': {
                 'readme_data': {'repo_name': 'fibonacci_series', 'extra_info': 'This is a fibonacci series repo.'}},
             'id': '2a0fbb62-a128-4698-8cd4-e17d7e438fd8', 'type': 'tool_call'}],
              usage_metadata={'input_tokens': 534, 'output_tokens': 35, 'total_tokens': 569,
                              'input_token_details': {'cache_read': 0}}),


    ToolMessage(
        content='{"success_messages": "Successfully created the README file for the fibonacci_series repository.", "file_content": "\\n    # fibonacci_series\\nThis is the README file for the fibonacci_series repository.\\n\\n        ## Prerequisite: \\n- Python 3.8+\\n\\n        ## Installation:\\n        \\n```bash\\n\\n        pip3 install -r requirements.txt\\n\\n        ```\\n\\n        ## Run code: \\n\\n        ```bash\\n\\n        python3 main.py\\n\\n        ```\\n        \\n##Details:\\nThis is a fibonacci series repo.\\n", "repo_name": "fibonacci_series"}',
        name='create_readme_file', id='c8f76849-2c3a-4929-a676-8c36ccc13cab',
        tool_call_id='2a0fbb62-a128-4698-8cd4-e17d7e438fd8'),


    AIMessage(content=[
        '```json\n{\n    "file_name": "README.md",\n    "file_path": "./README.md",\n    "file_content": "# fibonacci_series\\nThis is the README file for the fibonacci_series repository.\\n\\n        ## Prerequisite: \\n- Python 3.8+\\n\\n        ## Installation:\\n        \\n```',
        'bash\\n\\n        pip3 install -r requirements.txt\\n\\n        ',
        '```\\n\\n        ## Run code: \\n\\n        ```', 'bash\\n\\n        python3 main.py\\n\\n        ',
        '```\\n        \\n##Details:\\nThis is a fibonacci series repo."\n}\n```'], additional_kwargs={},
        response_metadata={
            'prompt_feedback': {'block_reason': 0,
                                'safety_ratings': []},
            'finish_reason': 'STOP', 'safety_ratings': []},
        id='run-0f565ec7-547f-4ada-81dc-92d1f471722a-0',
        usage_metadata={'input_tokens': 679,
                        'output_tokens': 153,
                        'total_tokens': 832,
                        'input_token_details': {
                            'cache_read': 0}}),


    AIMessage(content='', additional_kwargs={},
              response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP',
                                 'safety_ratings': []}, id='run-1ec4d838-99c2-403f-bc9e-67757f1f6cac-0',
              usage_metadata={'input_tokens': 796, 'output_tokens': 0, 'total_tokens': 796,
                              'input_token_details': {'cache_read': 0}})
]
