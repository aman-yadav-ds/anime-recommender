import joblib
import comet_ml
import os
import numpy as np
from typing import Dict, Tuple, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from src.base_model import BaseModel
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, data_path: str,
                  processed_data: Optional[Dict[str, Any]] = None, 
                 enable_comet: bool = False):
        """
        Args:
            data_path: Path to processed data directory
            processed_data: Optional pre-loaded processed data from DataProcessor
            enable_comet: Enable Comet ML logging
        """
        self.data_path = data_path
        self.processed_data = processed_data or {}
        self.enable_comet = enable_comet
        
        self.experiment = None
        if self.enable_comet:
            self._init_comet_experiment()
        
        self._configure_tensorflow()
        
        self._create_directories()
        
        logger.info(f"Model training initialized.")
    
    def _init_comet_experiment(self):
        try:
            self.experiment = comet_ml.Experiment(
                api_key=os.getenv('COMET_API_KEY'),
                project_name='anime_recommender',
                workspace='aman-yadav-ds'
            )
            logger.info("Comet ML experiment initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Comet ML: {e}")
            self.enable_comet = False
    
    def _configure_tensorflow(self):

        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all cores
        tf.config.threading.set_intra_op_parallelism_threads(0)
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        logger.info(f"TensorFlow configured.")
    
    def _create_directories(self):
        directories = [
            os.path.dirname(CHECKPOINT_FILE_PATH),
            MODEL_DIR,
            WEIGHTS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_data(self) -> Tuple[list, list, np.ndarray, np.ndarray, int, int]:
        """
        Load training data
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, n_users, n_anime)
        """
        try:
            # What we did with data ingestion step
            if self.processed_data:
                logger.info("Using pre-loaded processed data")
                X_train_array = self.processed_data['X_train_array']
                X_test_array = self.processed_data['X_test_array']
                y_train = self.processed_data['y_train']
                y_test = self.processed_data['y_test']
                
                n_users = len(self.processed_data['user2user_encoded'])
                n_anime = len(self.processed_data['anime2anime_encoded'])
                
            else:
                logger.info("Loading data from artifacts")
                X_train_array = joblib.load(X_TRAIN_ARRAY)
                X_test_array = joblib.load(X_TEST_ARRAY)
                y_train = joblib.load(Y_TRAIN)
                y_test = joblib.load(Y_TEST)
                
                n_users = len(joblib.load(USER2USER_ENCODED))
                n_anime = len(joblib.load(ANIME2ANIME_ENCODED))
            
            # Convert to optimal dtypes for training
            X_train_array = [arr.astype(np.int32) for arr in X_train_array]
            X_test_array = [arr.astype(np.int32) for arr in X_test_array]
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
            
            logger.info(f"Data loaded - Users: {n_users}, Anime: {n_anime}, "
                       f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
            
            return X_train_array, X_test_array, y_train, y_test, n_users, n_anime
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise CustomException("Failed to load training data", str(e))

    
    def create_callbacks(self, total_steps: int) -> list:
        """Create optimized callbacks for training"""
        callbacks = []
        
        checkpoint = ModelCheckpoint(
            filepath=CHECKPOINT_FILE_PATH,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=0
        )
        callbacks.append(checkpoint)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-4,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train the recommendation model
        
        Args:
            config_overrides: Optional configuration overrides
            
        Returns:
            Training results and model artifacts
        """
        try:
            X_train, X_test, y_train, y_test, n_users, n_anime = self.load_data()
            
            base_model = BaseModel(CONFIG_PATH)
            model = base_model.RecommenderNet(n_users, n_anime)
            
            config = {
                'batch_size': 10000,
                'epochs': 20,
                'learning_rate': 0.001,
            }
            
            total_steps = len(y_train) // config['batch_size'] * config['epochs']
            
            callbacks = self.create_callbacks(total_steps)
            
            if self.experiment:
                self.experiment.log_parameters(config)
                self.experiment.log_parameter("n_users", n_users)
                self.experiment.log_parameter("n_anime", n_anime)
            
            logger.info(f"Starting model training with config: {config}")
            
            # Train model
            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                verbose=1,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                shuffle=True
            )
            
            model.load_weights(CHECKPOINT_FILE_PATH)
            logger.info("Model training completed and best weights loaded")
            
            # Log training metrics to Comet
            if self.experiment:
                self._log_training_metrics(history)
            
            training_results = self.save_model_and_weights(model)
            training_results['history'] = history.history
            training_results['model'] = model
            
            logger.info("Training pipeline completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise CustomException("Model training failed", str(e))
    
    def _log_training_metrics(self, history):
        """Log training metrics to Comet ML"""
        try:
            for epoch in range(len(history.history['loss'])):
                metrics = {
                    'train_loss': history.history['loss'][epoch],
                    'val_loss': history.history['val_loss'][epoch],
                    'train_mae': history.history['mae'][epoch],
                    'val_mae': history.history['val_mae'][epoch],
                    'train_mse': history.history['mse'][epoch],
                    'val_mse': history.history['val_mse'][epoch],
                }
                
                for metric_name, metric_value in metrics.items():
                    self.experiment.log_metric(metric_name, metric_value, step=epoch)
                    
            logger.info("Training metrics logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log metrics to Comet ML: {e}")
    
    def extract_weights(self, model, layer_name: str) -> np.ndarray:
        """
        Extract and normalize embedding weights
        
        Args:
            model: Trained model
            layer_name: Name of embedding layer
            
        Returns:
            Normalized embedding weights
        """
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0].astype(np.float32)
            
            logger.info(f"Extracted and normalized weights for {layer_name}: {weights.shape}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to extract weights for {layer_name}: {str(e)}")
            raise CustomException(f"Weight extraction failed for {layer_name}", str(e))
    
    def save_model_and_weights(self, model) -> Dict[str, Any]:
        """
        Save model and extract embedding weights
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary containing saved artifacts
        """
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            
            user_weights = self.extract_weights(model, "user_embedding")
            anime_weights = self.extract_weights(model, "anime_embedding")
            
            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)
            
            if self.experiment:
                self.experiment.log_asset(MODEL_PATH)
                self.experiment.log_asset(USER_WEIGHTS_PATH)
                self.experiment.log_asset(ANIME_WEIGHTS_PATH)
            
            logger.info("Model and embedding weights saved successfully")
            
            return {
                'model_path': MODEL_PATH,
                'user_weights': user_weights,
                'anime_weights': anime_weights,
                'user_weights_path': USER_WEIGHTS_PATH,
                'anime_weights_path': ANIME_WEIGHTS_PATH
            }
            
        except Exception as e:
            logger.error(f"Failed to save model and weights: {str(e)}")
            raise CustomException("Model saving failed", str(e))
    
    def __del__(self):
        """Clean up resources"""
        if self.experiment:
            try:
                self.experiment.end()
            except:
                pass

def main():
    try:

        trainer = ModelTraining(
            data_path=PROCESSED_DIR,
            enable_comet=False,
        )
        
        results = trainer.train_model()
        
        print("\n=== Training Summary ===")
        final_train_loss = results['history']['loss'][-1]
        final_val_loss = results['history']['val_loss'][-1]
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"User Embeddings Shape: {results['user_weights'].shape}")
        print(f"Anime Embeddings Shape: {results['anime_weights'].shape}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()