from sentimentAnalyzer.pipeline.stage_01 import DataIngestionTrainingPipeline
from sentimentAnalyzer.pipeline.stage_02 import DataValidationTrainingPipeline
from sentimentAnalyzer.pipeline.stage_03 import DataTransformationTrainingPipeline
from sentimentAnalyzer.pipeline.stage_04 import ModelTrainingPipeline
from sentimentAnalyzer.logging import logger



STAGE_NAME = "Data Ingestion Stage"
try:
   logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data Validation Stage"
try:
   logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data transformation Stage"
try:
   logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Training Stage"
try:
   logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
   model_training = ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e