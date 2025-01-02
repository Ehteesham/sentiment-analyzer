from sentimentAnalyzer.constant import *
from sentimentAnalyzer.utils.common import read_yaml, create_directories
from sentimentAnalyzer.entity import DataIngestionConfig

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
