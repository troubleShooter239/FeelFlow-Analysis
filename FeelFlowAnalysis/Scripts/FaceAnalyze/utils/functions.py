from os import getenv
from pathlib import Path

def get_deepface_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory."""
    return str(getenv("DEEPFACE_HOME", default=str(Path.home())))