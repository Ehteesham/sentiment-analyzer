import joblib
from sentimentAnalyzer.logging import logger
from sentimentAnalyzer.config.configuration import ConfigurationManager
from sentimentAnalyzer.component.data_transformation import DataTransformation
from sentimentAnalyzer.entity import UserPredictionConfig


class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.config = config.get_user_prediction_config()
        data_transformation_config = config.get_data_transformation_config()
        self.data_transformation = DataTransformation(config=data_transformation_config)

    def predict(self, text):
        lrmodel = joblib.load(self.config.model_dir)
        logger.info(f"Trained Model Loaded from {self.config.model_dir}")

        # Text Pre-processing
        text = self.data_transformation.lower_case_converter(text)
        text = self.data_transformation.basic_remove_process(text)
        text = self.data_transformation.text_tokenizer(text)
        text = self.data_transformation.stop_word_removal(text)
        text = self.data_transformation.text_stemmer(text)

        text_func = lambda text: ' '.join(text) if isinstance(text, list) else text
        text = text_func(text)

        # Loading the Vectoriser
        vectoriser = joblib.load(self.config.vectoriser_dir)
        transformed_text = vectoriser.transform([text])
        logger.info(f"User Text Pre-processed Successfully")

        final_output = lrmodel.predict(transformed_text)

        # Returing the Sentiment
        sentiment = ""
        if final_output == 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Positive'

        return sentiment

    