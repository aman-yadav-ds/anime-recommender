from src.data_processing import DataProcessor  
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml

if __name__ == '__main__':
    config = read_yaml(CONFIG_PATH)

    # Step 1: Process data  
    processor = DataProcessor(PROCESSED_DIR)
    processed_data = processor.process_all_data()

    # Step 2: Train model with processed data
    trainer = ModelTraining(
        data_path=PROCESSED_DIR,
        processed_data=processed_data,
        enable_comet=True
    )

    trainer.train_model()