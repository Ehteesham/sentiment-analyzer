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

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    file_check_dir: Path
    FILE_NAMES: list


@dataclass(frozen=True)
class DataTransformationConfig:
    dataset_dir: Path
    transformed_data_dir: Path
    train_data_file: Path
    test_data_file: Path
    train_transformed_dir: Path
    test_transformed_dir: Path
    encoder: str
    max_features: int
    ngram_range: tuple


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_dir: Path
    train_data_file: Path
    # Model Parameters
    C: float
    max_iter: int
    n_jobs: int
    penalty: str
    solver: str
    class_weight: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_transformed_dir: Path
    model_dir: Path
    evaluation_saved: Path