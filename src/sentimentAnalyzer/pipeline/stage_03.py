from sentimentAnalyzer.config.configuration import ConfigurationManager
from sentimentAnalyzer.component.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_validation = DataTransformation(config=data_transformation_config)
        data_validation.data_transformation()