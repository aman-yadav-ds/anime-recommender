# Seamless integration with previous steps
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor  
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml

# Complete pipeline
config = read_yaml(CONFIG_PATH)

# Step 1: Ingest data
ingestor = DataIngestion(config)
dataframes = ingestor.download_from_gcp()

# Step 2: Process data  
processor = DataProcessor(PROCESSED_DIR, input_dataframes=dataframes)
processed_data = processor.process_all_data()

# Step 3: Train model with processed data
trainer = ModelTraining(
    data_path=PROCESSED_DIR,
    processed_data=processed_data,  # No file I/O needed!
    enable_comet=True
)

results = trainer.train_model()

# Use trained model immediately
model = results['model']
user_embeddings = results['user_weights']