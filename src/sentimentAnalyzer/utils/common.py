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
def load_transformed_data_file(path: Path, data_info: DataInfo) -> tuple:
    """
    Load transformed input and output data from files in a directory.

    Args:
        path (Path): Path to the directory containing transformed data files.
        data_info (DataInfo): Metadata about whether this data is for training or testing.

    Returns:
        dict: A dictionary with keys "input" (csr_matrix) and "output" (np.ndarray).
              Example:
              {
                  "input": <csr_matrix>,
                  "output": <ndarray>
              }

    Raises:
        FileNotFoundError: If the directory does not contain any valid files.
        ValueError: If a file has an unsupported extension.

    Notes:
        - The function expects at least one `.npz` file (for input data) and one `.npy` file
          (for output data) in the provided directory.
        - If the directory is empty, an exception is raised.
        - Unsupported file extensions will result in a ValueError.
    """

    path_file_lst = list(path.glob("*"))
    result_dic = dict()

    if not path_file_lst:
        logger.info(f"{path_file_lst} does not exist")
        raise FileNotFoundError(f"File not found: {path_file_lst}")
    
    for file_path in path_file_lst:
        
        if file_path.suffix == ".npz":
            data = load_npz(file_path)
            result_dic["input"] = data
            logger.info(f"Input {data_info.value} has been load from {file_path}")
        elif file_path.suffix == ".npy":
            data = np.load(file_path)
            result_dic['output'] = data
            logger.info(f"Output {data_info.value} is load from {file_path}")
        else:
            logger.info(f"{file_path} file extension not supported")
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")
    
    return result_dic.get("input"), result_dic.get("output")