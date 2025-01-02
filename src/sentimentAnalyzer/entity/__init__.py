from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_dir: Path
    unzip_dir: Path
    dataset_file: Path
    train_data_dir: Path
    test_data_dir: Path
    encoding: str
    train_size: int
    test_size: int