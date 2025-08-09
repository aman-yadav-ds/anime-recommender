import os
import pandas as pd
from google.cloud import storage
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

        logger.info("Data Ingestion started...")
    
    def download_from_gcp(self):
        try:
            logger.info("Downloading CSV files from Google Bucket....")

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                blob = bucket.blob(file_name)

                if file_name == "animelist.csv":
                    logger.info("Large File Detected. Directly reading first 5M rows using GCS blob as file-like object...")
                    # When we are downloading data from gcp bucket we are downloading the whole datasest(2GB) only to 
                    # discard 65M rows which will incur the cost of downloading 2GB of data from the bucket as a single operation.
                    # Let's say this charge(download(2GB)+ One read Operation) is charge_2GB.

                    # But if we are to stream the file using the code below:

                    # {Code:}
                    # blob = bucket.blob(file_name) //animelist.csv here
                    # with blob.open("r", encoding='utf-8') as gcs_file:
                    #             data = pd.read_csv(gcs_file, nrows=5_000_000)
                    # data.to_csv(file_path, index=False)

                    # we would only be paying the charge for downloading data of the size equivalent to (5M rows).
                    # While we will be making many Read HTTP request or you can say read operations.
                    # The cost for these operation in comparision to downloading the whole 2GB of data will 
                    # be negligible and lets say this cost(download(5M rows) + (many read operations)) as charge_5M.

                    # Then This charge_5M will be much less than charge_2GB. And additionally we won't have to load
                    # the full 2GB data into memory like we are doing when we download the 2GB
                    # data first which will reduce our compute resources. This will also benefit us
                    # when we containarize our app and use GKE later.

                    with blob.open("r", encoding='utf-8') as gcs_file:
                        data = pd.read_csv(gcs_file, nrows=1_000_000)
                        
                    data.to_csv(file_path, index=False)

                    logger.info(f"Downloaded and saved first 5M rows of '{file_name}' to '{file_path}'.")
                else:
                    logger.info(f"Downloading Smaller file {file_name}...")
                    blob.download_to_filename(file_path)
                    logger.info(f"Downloaded '{file_name}' to '{file_path}'.")
            
            logger.info("Data Ingestion Done.")
        except CustomException as ce:
            logger.error("Error while downloading data from GCP.")
            raise CustomException("Custom Exception", str(ce))

if __name__ == "__main__":
    ingestor = DataIngestion(read_yaml(CONFIG_PATH))
    ingestor.download_from_gcp()