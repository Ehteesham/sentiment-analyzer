import os
from box.exceptions import BoxValueError
import yaml
from sentimentAnalyzer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import pandas as pd



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



@ensure_annotations
def read_dataset(path: Path, encoding: str) -> pd.DataFrame:
    """
    Reads a CSV dataset into a pandas DataFrame.

    Args:
        path (Path): The file path to the dataset.
        encoding (str): The encoding format for the file. Defaults to 'utf-8'.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file at the given path does not exist.
        ValueError: If an error occurs during file reading.
    """
    cols_name = ["target", "ids", "date", "flag", "user", "text"]
    if not os.path.exists(path):
        logger.error(f"File not found at path: {path}")
        raise FileNotFoundError(f"The file at {path} does not exist.")
    
    try:
        logger.info(f"Reading dataset from path: {path}")
        df = pd.read_csv(path, encoding=encoding, names=cols_name)
        logger.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Error occurred while reading the dataset: {e}")
        raise ValueError(f"Failed to read the dataset at {path}. Error: {e}")