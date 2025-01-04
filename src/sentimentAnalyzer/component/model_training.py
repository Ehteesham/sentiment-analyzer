import joblib
from sklearn.linear_model import LogisticRegression
from sentimentAnalyzer.logging import logger
from sentimentAnalyzer.entity import ModelTrainingConfig
from sentimentAnalyzer.utils.common import load_transformed_data_file, DataInfo



class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def train_model(self):
        X_train, y_train = load_transformed_data_file(path=self.config.train_data_file, 
                                               data_info=DataInfo.TRAINING)
        
        X_test, y_test = load_transformed_data_file(path=self.config.test_data_file,
                                                    data_info=DataInfo.TESTING)
        
        # Getting Model Parameters
        C = self.config.C
        max_iter = self.config.max_iter
        n_jobs = self.config.n_jobs
        penalty = self.config.penalty
        solver = self.config.solver
        class_weight = self.config.class_weight


        lrmodel = LogisticRegression(C = C, 
                                     max_iter=max_iter, 
                                     n_jobs=n_jobs, 
                                     penalty=penalty, 
                                     solver=solver, 
                                     class_weight=class_weight)
        # Model Training
        lrmodel.fit(X_train, y_train)
        logger.info("Model Training Completed")
        
        # Saving the Model
        if self.config.model_dir.exists():
            joblib.dump(lrmodel, f"{self.config.model_dir}/trained_model.pkl")
            logger.info(f'Trained Model is Saved in {self.config.model_dir}')
        else:
            logger.info("Directory not found and model not Saved")
            raise FileNotFoundError("Directory is Not Found")