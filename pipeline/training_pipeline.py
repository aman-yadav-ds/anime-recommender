from src.data_processing import DataProcessor  
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml

if __name__ == '__main__':
    config = read_yaml(CONFIG_PATH)

    processor = DataProcessor(PROCESSED_DIR)
    processed_data = processor.process_all_data(input_file=ANIMELIST_CSV)

    trainer = ModelTraining(
        data_path=PROCESSED_DIR,
        processed_data=processed_data,
        enable_comet=True
    )

    trainer.train_model()