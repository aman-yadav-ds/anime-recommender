import os
import pandas as pd
from google.cloud import storage
from typing import Dict
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config['bucket_name']
        self.file_names = self.config['bucket_file_names']

        os.makedirs(RAW_DIR, exist_ok=True)

        # for protection so they are not used outside of this class
        self._client = storage.Client()
        self._bucket = self._client.bucket(self.bucket_name)

        logger.info("Data Ingestion Initialized...")

    def download_from_gcp(self) -> Dict[str, pd.DataFrame]:
        """
        Download csv files from GCS bucket

        Args:
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dict mapping file names to their DataFrames
        """
        try:
            logger.info("Downloading CSV files from Google Bucket....")

            dataframes = {}



            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                blob = self._bucket.blob(file_name)

                if file_name == "animelist.csv":
                    logger.info(f"Streaming Large file for donwload: {file_name} (first 8M rows)")


                    with blob.open("r", encoding="utf-8") as gcs_file:
                        data = pd.read_csv(
                            gcs_file,
                            nrows=5_000_000,
                            low_memory=False  #We need speed for this operation rather than saving memory
                        )

                    data.to_csv(file_path, index=False)
                    dataframes[file_name] = data
                    logger.info(f"Saved {len(data)} rows from {file_name} to {file_path}")

                else:
                    logger.info(f"Downloading smaller files {file_name}...")


                    with blob.open("r", encoding='utf-8') as gcs_file:
                        data = pd.read_csv(
                            gcs_file,
                            low_memory=False
                        )

                    data.to_csv(file_path, index=False)
                    dataframes[file_name] = data
                    logger.info(f"Saved {len(data)} rows from {file_name} to {file_path}")
                                
            logger.info(f"Data Ingestion completed.")
            return dataframes
        except Exception as e:
            logger.error("Error during data ingestion from GCP.")
            raise CustomException("Data ingestion failed", str(e))
        
    def __del__(self):
        """
        Clean up resources because we don't want our 
        storage client open anymore after it has done its work.
        """
        if hasattr(self, '_client'):
            self._client.close()


def main():

    config = read_yaml(CONFIG_PATH)
    ingestor = DataIngestion(config)
    dataframes = ingestor.download_from_gcp()

    for file_name, df in dataframes.items():
        print(f"{file_name}: {df.shape}")

    return dataframes

if __name__ == "__main__":
    main()