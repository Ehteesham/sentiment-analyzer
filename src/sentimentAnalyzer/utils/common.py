import os
from enum import Enum
from box.exceptions import BoxValueError
import yaml
from sentimentAnalyzer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Union
import pandas as pd
import numpy as np
from scipy.sparse import save_npz, load_npz, csr_matrix

class DataInfo(str ,Enum):
    TRAINING = "Transformed Training Data"
    TESTING = "Transformed Testing Data"



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
def read_dataset(path: Path, encoding: str, training=False) -> pd.DataFrame:
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
        if os.path.exists(path) and training==True:
            df = pd.read_csv(path, encoding=encoding, names=cols_name)
        else:
            df = pd.read_csv(path, encoding=encoding)
        logger.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Error occurred while reading the dataset: {e}")
        raise ValueError(f"Failed to read the dataset at {path}. Error: {e}")
    
@ensure_annotations
def save_transformed_data_file(path: Path, data, data_info: DataInfo):
    """
    Saves transformed data to the specified file path with proper format and logs the process.

    Args:
        path (Path): The file path where the data should be saved.
        data: The data to be saved. Can be sparse or dense, depending on the format.
        data_info (DataInfo): Enum indicating if the data is for training or testing.

    Returns:
        None
    """
    if path.suffix == ".npz":
        save_npz(path, data)
        logger.info(f"{data_info.value} is Saved in {path}")
    elif path.suffix == ".npy":
        np.save(path, data)
        logger.info(f"{data_info.value} is Saved in {path}")
    else:
        logger.info(f"{path.suffix} is not a correct extension {data_info.value} will not be stored")


@ensure_annotations
def load_transformed_data_file(path: Path, data_info: DataInfo) -> Union[np.ndarray, csr_matrix]:
    """
    Load transformed data from file.

    Args:
        path (Path): Path to the file to load.
        data_info (DataInfo): Metadata about whether this is training or testing data.

    Returns:
        Union[np.ndarray, csr_matrix]: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """

    if not path.exists():
        logger.info(f"{path} does not exist")
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix == ".npz":
        data = load_npz(path)
        logger.info(f"{data_info.value} is load from {path}")
    elif path.suffix == ".npy":
        data = np.load(path)
        logger.info(f"{data_info.value} is load from {path}")
    else:
        logger.info(f"{path} file extension not supported")
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    
    return data