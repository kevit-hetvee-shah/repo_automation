import os
import sys
from typing import Type


def from_file(file_path: str):
    """Dynamically imports a BaseTool class from a Python file within a package structure.

    Parameters:
        file_path: The file path to the Python file containing the BaseTool class.

    Returns:
        The imported BaseTool class.
    """
    file_path = os.path.relpath(file_path)
    # Normalize the file path to be absolute and extract components
    directory, file_name = os.path.split(file_path)
    import_path = os.path.splitext(file_path)[0].replace(os.sep, ".")
    class_name = os.path.splitext(file_name)[0]

    exec_globals = globals()

    current_working_directory = os.getcwd()
    sys.path.append(current_working_directory)
    exec(f"from {import_path} import {class_name}", exec_globals)

    imported_class = exec_globals.get(class_name)
    if not imported_class:
        raise ImportError(f"Could not import {class_name} from {import_path}")

    return imported_class
