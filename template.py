import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "sentimentAnalyzer"

list_of_files = [
    "app.py",
    "main.py",
    "setup.py",
    "requirements.txt",
    "params.yaml",
    "Dockerfile",
    "config/config.yaml",
    "notebooks/trial.ipynb",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/component/__init__.py",
    f"src/{project_name}/constant/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/logging/__init__.py",
]


for files in list_of_files:
    files = Path(files)
    files_dir, files_name = os.path.split(files)

    if files_dir != "":
        os.makedirs(files_dir, exist_ok=True)
        logging.info(f"Directory is Created: {files_dir}")

    if (not os.path.exists(files)) or (os.path.getsize(files)==0):
        with open(files, "w") as f:
            pass

        logging.info(f"Files created|: {files_name}")
    else:
        logging.info(f"File Already Exists: {files}")
