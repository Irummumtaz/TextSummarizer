from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainerTrainingPipeline

STAGE_NAME='Data_Ingestion Stage'

try:
    logger.info(f"stage{STAGE_NAME} Initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f'Stage Ingestion {STAGE_NAME} completed')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME='Data Tranformation Stage'

try:
    logger.info(f"stage{STAGE_NAME} Initiated")
    data_ingestion_pipeline=DataTransformationTrainingPipeline()
    data_ingestion_pipeline.initiate_data_transformation()
    logger.info(f'Stage Ingestion {STAGE_NAME} completed')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME='Model Trainer Stage'

try:
    logger.info(f"stage{STAGE_NAME} Initiated")
    model_trainer_pipeline=ModelTrainerTrainingPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f'Stage Ingestion {STAGE_NAME} completed')
except Exception as e:
    logger.exception(e)
    raise e