from sentimentAnalyzer.config.configuration import ConfigurationManager
from sentimentAnalyzer.component.model_training import ModelTraining

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.train_model()