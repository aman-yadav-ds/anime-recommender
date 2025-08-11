import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Optional, Any, List
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, output_dir: str):
        """
        Initialize DataProcessor with optional pre-loaded DataFrames
        
        Args:
            output_dir: Directory to save processed data
            input_dataframes: Optional dict of DataFrames from data ingestion
        """
        self.output_dir = output_dir
    
        # Initialize data containers
        self.rating_df: Optional[pd.DataFrame] = None
        self.anime_df: Optional[pd.DataFrame] = None
        
        # Training data containers
        self.X_train_array: Optional[list] = None
        self.X_test_array: Optional[list] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.encoders = {
            'user2user_encoded': {},
            'user2user_decoded': {},
            'anime2anime_encoded': {},
            'anime2anime_decoded': {}
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Data Processing initialized.")

    def load_data(self, usecols: List, input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Try to use passed DataFrame first
        
        Args:
            usecols: Columns to load
            input_file: Fallback file path if DataFrame not passed
            
        Returns:
            DataFrame
        """
        try:
            logger.info(f"Loading data from file: {input_file}")
            self.rating_df = pd.read_csv(input_file, usecols=usecols, low_memory=False)
            
            logger.info(f"Data loaded successfully. Shape: {self.rating_df.shape}")
            return self.rating_df
        except FileNotFoundError:
            raise FileNotFoundError(f"DataFrame not provided and file {input_file} not found.")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise CustomException("Failed to load data", str(e))
        
        
    # def filter_users(self, min_rated: int = 400) -> pd.DataFrame:
    #     """Filter users based on if they rated more than 400 animes"""
    #     try:
    #         logger.info(f"Filtering users with min_rated >= {min_rated}")
            
            
    #         user_counts = self.rating_df['user_id'].value_counts()
    #         valid_users = user_counts[user_counts >= min_rated].index
            
    #         initial_shape = self.rating_df.shape[0]
    #         self.rating_df = self.rating_df[self.rating_df['user_id'].isin(valid_users)]
            
    #         logger.info(f"Filtered users: {initial_shape} -> {self.rating_df.shape[0]} rows")
    #         return self.rating_df
            
    #     except Exception as e:
    #         logger.error(f"Failed to filter users: {str(e)}")
    #         raise CustomException("Failed to filter users", str(e))
        
    def scale_ratings(self) -> pd.DataFrame:
        """sacling ratings to range -> [0, 1]."""
        try:
            min_rating = np.min(self.rating_df['rating'])
            max_rating = np.max(self.rating_df['rating'])

            self.rating_df['rating'] = ((self.rating_df['rating'] - min_rating) / max_rating - min_rating)
            
            logger.info(f"Scaling ratings completed.")
            return self.rating_df
            
        except Exception as e:
            logger.error(f"Failed to scale ratings: {str(e)}")
            raise CustomException("Failed to scale ratings", str(e))
        
    def encode_data(self) -> Dict[str, Dict]:
        """Encoding ids into series for our model"""
        try:
           
            user_ids = self.rating_df['user_id'].unique()
            anime_ids = self.rating_df['anime_id'].unique()
            
           
            self.encoders['user2user_encoded'] = {uid: idx for idx, uid in enumerate(user_ids)}
            self.encoders['user2user_decoded'] = {idx: uid for idx, uid in enumerate(user_ids)}
            self.encoders['anime2anime_encoded'] = {aid: idx for idx, aid in enumerate(anime_ids)}
            self.encoders['anime2anime_decoded'] = {idx: aid for idx, aid in enumerate(anime_ids)}
            
            
            self.rating_df['user'] = self.rating_df['user_id'].map(self.encoders['user2user_encoded']).astype('uint16') #this is just for memory efficiency
            self.rating_df['anime'] = self.rating_df['anime_id'].map(self.encoders['anime2anime_encoded']).astype('uint16')
            
            logger.info(f"Encoding completed. Users: {len(user_ids)}, Animes: {len(anime_ids)}")
            return self.encoders
            
        except Exception as e:
            logger.error(f"Failed to encode data: {str(e)}")
            raise CustomException("Failed to encode data", str(e))
        
    def split_data(self, test_size: int = 1000, random_state: int = 42) -> Tuple[list, list, np.ndarray, np.ndarray]:
        """Split data. we only use 1000 instance for validation set"""
        try:
            # Sample for randomization
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            X = self.rating_df[['user', 'anime']].values

            # To train our model on binary_crossentropy
            y = self.rating_df['rating'].apply(lambda x: 0 if x<0.6 else 1).to_numpy()
            
            train_size = len(X) - test_size
            
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            self.y_train, self.y_test = y_train, y_test
            
            # Convert to list because we pass two inputs to our model -[user, anime]
            self.X_train_array = [X_train[:, 0], X_train[:, 1]]
            self.X_test_array = [X_test[:, 0], X_test[:, 1]]
            
            logger.info(f"Data split completed. Train: {len(self.y_train)}, Test: {len(self.y_test)}")
            return self.X_train_array, self.X_test_array, self.y_train, self.y_test
            
        except Exception as e:
            logger.error(f"Failed to split data: {str(e)}")
            raise CustomException("Failed to split data", str(e))
        
    def save_artifacts(self) -> Dict[str, Any]:
        """Save all artifacts and return them for model training."""
        try:
            saved_artifacts = {}
            
            for name, encoder in self.encoders.items():
                file_path = os.path.join(self.output_dir, f"{name}.pkl")
                joblib.dump(encoder, file_path)
                saved_artifacts[name] = encoder
                logger.info(f"Saved {name} to {file_path}")
            
            training_data = {
                'X_train_array': (X_TRAIN_ARRAY, self.X_train_array),
                'X_test_array': (X_TEST_ARRAY, self.X_test_array),
                'y_train': (Y_TRAIN, self.y_train),
                'y_test': (Y_TEST, self.y_test)
            }
            
            for name, (path, data) in training_data.items():
                joblib.dump(data, path)
                saved_artifacts[name] = data
                logger.info(f"Saved {name} to {path}")
            
            self.rating_df.to_csv(RATING_DF, index=False)
            saved_artifacts['rating_df'] = self.rating_df
            logger.info(f"Saved rating_df to {RATING_DF}")
            
            logger.info("All artifacts saved successfully")
            return saved_artifacts
            
        except Exception as e:
            logger.error(f"Failed to save artifacts: {str(e)}")
            raise CustomException("Failed to save artifacts", str(e))
        
    def process_anime_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process anime data with optimizations"""
        try:

            df = pd.read_csv(ANIME_CSV)
            logger.info(f"Loaded anime data from file: {ANIME_CSV}")
            
            synopsis_cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            synopsis_df = pd.read_csv(ANIME_SYNOPSIS_CSV, usecols=synopsis_cols)
            
            df = df.replace("Unknown", np.nan)
            
            def get_anime_name(df):
                mask_eng = df['eng_version'].notna()
                df.loc[mask_eng, 'final_name'] = df.loc[mask_eng, 'eng_version']
                df.loc[~mask_eng, 'final_name'] = df.loc[~mask_eng, 'Name']
                return df['final_name']
            
            df['anime_id'] = df['MAL_ID']
            df['eng_version'] = df['English name']
            
            df['eng_version'] = get_anime_name(df)
            
            df = df.sort_values(by='Score', ascending=False, na_position='last', kind='mergesort')
            
            final_cols = ["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Members", "Premiered"]
            df = df[final_cols].copy()
            
            # Save processed data
            df.to_csv(ANIME_DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)
            
            logger.info(f"Processed anime data: {df.shape[0]} records")
            return df, synopsis_df
            
        except Exception as e:
            logger.error(f"Failed to process anime data: {str(e)}")
            raise CustomException("Failed to process anime data", str(e))

    def process_all_data(self, input_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete processing pipeline and return all processed data
        
        Args:
            input_file: Fallback file path if DataFrames not provided
            
        Returns:
            Dictionary containing all processed data
        """
        try:
            logger.info("Starting optimized data processing pipeline")
            
            self.load_data(usecols=["user_id", "anime_id", "rating"], input_file=input_file)
            # self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            
            # Save artifacts and get them back for return
            processed_data = self.save_artifacts()
            
            # Process anime data
            anime_df, synopsis_df = self.process_anime_data()
            processed_data.update({
                'anime_df': anime_df,
                'synopsis_df': synopsis_df
            })
            
            logger.info("Data processing pipeline completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise CustomException("Data processing pipeline failed", str(e))

def main():
    try:
        # We won't use this pipeline this way in our full pipeline but we'll pass the data we'll get from ingestion step.
        data_processor = DataProcessor(PROCESSED_DIR)
        processed_data = data_processor.process_all_data(input_file=ANIMELIST_CSV)
        
        print("\n=== Processing Summary ===")
        for key, value in processed_data.items():
            if isinstance(value, pd.DataFrame):
                print(f"{key}: {value.shape}")
            elif isinstance(value, (list, dict)):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {type(value)}")
                
        return processed_data
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()