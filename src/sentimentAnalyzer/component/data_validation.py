import os
from sentimentAnalyzer.logging import logger
from sentimentAnalyzer.entity import DataValidationConfig

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_folders(self) -> None:
        try:
            all_files = os.listdir(self.config.file_check_dir)
            flag = None
            status_file_path = self.config.STATUS_FILE
            for file in all_files:
                if file not in self.config.FILE_NAMES:
                    flag = False
                    with open(status_file_path, "w") as f:
                        f.write(f"Validatio Status: {flag}")
                else:
                    flag = True
                    with open(status_file_path, "w") as f:
                        f.write(f"Validatio Status: {flag}")
            return None
        except Exception as e:
            raise e