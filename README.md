.dvc/
    ├── .gitignore
    └── config
artifacts/
    ├── .gitignore
    ├── model_checkpoints/
    |    └── weights.weights.h5
    ├── model/
    |    └── model.h5
    ├── processed/
    |    ├── anime_df.csv
    |    ├── rating_df.csv
    |    ├── synopsis.csv
    |    ├── anime2anime_encoded.pkl
    |    ├── anime2anime_decoded.pkl
    |    ├── user2user_encoded.pkl
    |    ├── user2user_decoded.pkl
    |    ├── X_train_array.pkl
    |    ├── X_test_array.pkl
    |    ├── y_train.pkl
    |    ├── y_test.pkl
    ├── raw/
    |    ├── anime_with_synopsis.csv
    |    ├── animelist.csv
    |    └── anime.csv
    └── weights/
    |    ├── anime_weights.pkl
    |    └── user_weights.pkl
    ├── model_checkpoints.dvc
    ├── model.dvc
    ├── processed.dvc
    ├── raw.dvc
    └── weights.dvc
config/
    ├── __init__.py
    ├── config.yaml
    └── paths_config.py
notebook/
    └── anime.ipynb
pipeline/
    ├── __init__.py
    ├── prediction_pipeline.py
    └── training_pipeline.py
src/
    ├── __init__.py
    ├── base_model.py
    ├── custom_exception.py
    ├── data_ingestion.py
    ├── data_processing.py
    ├── logger.py
    └── model_training.py
static/
    ├── script.js
    └── styles.css
templates/
    └── index.html
utils/
    ├── __init__.py
    ├── common_functions.py
    └── helpers.py
.dvcignore
.gitignore
application.py
Dockerfile
Jenkinsfile
LICENSE
pyproject.toml

[Jenkins] ---> [Pipeline Docker Image Build & Run] ---> [Artifacts in GCS or PersistentVolume]
                     |
                     v
             [Webapp Docker Image Build] ---> [Deploy to GKE]
