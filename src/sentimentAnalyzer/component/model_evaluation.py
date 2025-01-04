import joblib
import pandas as pd
from sentimentAnalyzer.utils.common import load_transformed_data_file, DataInfo
from sentimentAnalyzer.entity import ModelEvaluationConfig
from sentimentAnalyzer.logging import logger
from sklearn.metrics import (classification_report, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        X_test, y_test = load_transformed_data_file(path=self.config.test_transformed_dir, 
                                                    data_info=DataInfo.TESTING)
        
        # Loading Models
        lrmodel = joblib.load(self.config.model_dir)
        logger.info(f"Trained Model imported from {self.config.model_dir}")

        # Classification Report
        y_pred = lrmodel.predict(X_test)
        print(f">>>>>>> Classifiaction Report <<<<<<< \n{classification_report(y_test, y_pred)}")

        # Saving Evaluation Result
        metrices = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        }

        # Save metrics to a CSV
        metrics_df = pd.DataFrame(list(metrices.items()), columns=["Metric", "Value"])
        metrics_df.to_csv(self.config.evaluation_saved, index=False)
        logger.info(f"Model Performance Metrices stored at {self.config.evaluation_saved}")
