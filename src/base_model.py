from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path):
        try:
            self.config = read_yaml(config_path)
            self.embedding_size = self.config['model']['embedding_size']
            self.loss = self.config['model']['loss']
            self.optimizer = Adam(learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            self.metrics = self.config['model']['metrics']
            logger.info('Model Training Configuration Loaded.')
        except Exception as e:
            raise CustomException("Error Loading Model Training configurations", e)
    def RecommenderNet(self, n_users, n_anime):
        try:
            embedding_size = self.embedding_size

            user = Input(name="user",shape=[1])

            user_embedding = Embedding(name="user_embedding",input_dim=n_users,output_dim=embedding_size, embeddings_regularizer=l2(1e-6))(user)

            anime = Input(name="anime",shape=[1])

            anime_embedding = Embedding(name="anime_embedding",input_dim=n_anime,output_dim=embedding_size, embeddings_regularizer=l2(1e-6))(anime)

            x = Dot(name="dot_product" , normalize=True , axes=2)([user_embedding,anime_embedding])

            x = Flatten()(x)
            x = Dense(1,kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)

            x = Activation("sigmoid")(x)

            model = Model(inputs=[user,anime], outputs=x)
            
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metrics
            )
            logger.info("Model Created Succesfully...")
            return model
        except Exception as e:
            raise CustomException("Error while Compiling Base model.", e)