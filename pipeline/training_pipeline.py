from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor  
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml

if __name__ == '__main__':
    config = read_yaml(CONFIG_PATH)
    ingestor = DataIngestion(config)
    dataframes = ingestor.download_from_gcp()

    processor = DataProcessor(PROCESSED_DIR, input_dataframes=dataframes)
    processed_data = processor.process_all_data()

    trainer = ModelTraining(
        data_path=PROCESSED_DIR,
        processed_data=processed_data,
        enable_comet=True
    )

    trainer.train_model()