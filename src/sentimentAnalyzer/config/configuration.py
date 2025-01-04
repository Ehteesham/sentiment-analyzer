from sentimentAnalyzer.constant import *
from sentimentAnalyzer.utils.common import read_yaml, create_directories
from sentimentAnalyzer.entity import (DataIngestionConfig, 
                                      DataValidationConfig,
                                      DataTransformationConfig)

class ConfigurationManager:
    def __init__(self, config_path = CONFIG_FILE_PATH, params_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        dataset = self.params.dataset
        create_directories([config.root_dir])
        create_directories([config.train_data_dir])
        create_directories([config.test_data_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_dir = config.source_dir,
            unzip_dir = config.unzip_dir,
            dataset_file = config.dataset_file,
            train_data_dir = config.train_data_dir,
            test_data_dir = config.test_data_dir,
            encoding = dataset.data_encoding,
            train_size = dataset.train_size,
            test_size = dataset.test_size 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation = DataValidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            file_check_dir = config.file_check_dir,
            FILE_NAMES = config.FILE_NAMES
        )

        return data_validation
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.transformed_data_dir])
        create_directories([config.train_transformed_dir])
        create_directories([config.test_transformed_dir])

        data_transformation_config = DataTransformationConfig(
            dataset_dir = Path(config.dataset_dir),
            transformed_data_dir = Path(config.transformed_data_dir),
            train_data_file = Path(config.train_data_file),
            test_data_file = Path(config.test_data_file),
            train_transformed_dir = Path(config.train_transformed_dir),
            test_transformed_dir = Path(config.test_transformed_dir),
            encoder = self.params.dataset.data_encoding,
            max_features = self.params.text_vectoriser.max_features,
            ngram_range = self.params.text_vectoriser.max_features,
        )

        return data_transformation_config
