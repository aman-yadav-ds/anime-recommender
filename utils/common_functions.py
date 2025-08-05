import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        logger.info("Reading YAML file...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not found on given path")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read YAML file.")
            return config
    except Exception as e:
        logger.error("Error while reading YAML file.")
        raise CustomException("Failed to read YAML file.", e)
    
def load_data(path):
    try:
        logger.info("Loading data...")

        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error Loading Data {e}.")
        raise CustomException("Failed to load Data.", e)