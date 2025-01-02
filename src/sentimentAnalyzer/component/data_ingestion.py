import os
import zipfile
from pathlib import Path
from sentimentAnalyzer.entity import DataIngestionConfig
from sentimentAnalyzer.logging import logger
from sentimentAnalyzer.utils.common import read_dataset
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.source_dir, 'r') as zip_file:
            zip_file.extractall(unzip_path)
        
    def split_dataset(self):
        dataset_path = Path(self.config.dataset_file)
        encoding = self.config.encoding
        df = read_dataset(path=dataset_path, encoding=encoding)
        train_data, test_data = train_test_split(df,
                                                 train_size=self.config.train_size,
                                                 test_size=self.config.test_size,
                                                 random_state=42)

        train_data.to_csv(f"{self.config.train_data_dir}/train.csv", index = False)
        logger.info(f"Train Data has been stored at {self.config.train_data_dir}/train.csv")
        test_data.to_csv(f"{self.config.test_data_dir}/test.csv", index = False)
        logger.info(f"Test Data has been stored at {self.config.test_data_dir}/test.csv")
        
        
