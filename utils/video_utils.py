"""Video processing utilities"""

import os

def get_valid_path(prompt_message):
    """Get a valid file path from user input"""
    while True:
        path = input(prompt_message).strip()
        if path.lower() in ['exit', 'quit']:
            return None
        if os.path.exists(path) and os.path.isfile(path):
            return path
        else:
            print(f"Error: The path '{path}' does not exist or is not a valid file. "
                  "Please try again or type 'exit' to quit.")