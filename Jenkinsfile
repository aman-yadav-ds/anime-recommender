pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage("Cloning from Github....") {
            steps {
                script {
                    echo 'Cloning from Github.....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/aman-yadav-ds/anime-recommender.git']])
                }
            }
        }

        stage("Making a Virtual Environment....") {
            steps {
                script {
                    echo 'Making a Virtual Environment....'
                    sh """
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    pip install dvc
                    """
                }
            }
        }
        stage("DVC Pull....") {
            steps {
                withCredentials([file(credentialsId:'recommender-gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script {
                        echo "DVC Pull...."
                        sh """
                        . ${VENV_DIR}/bin/activate
                        dvc pull
                        """
                    }
                }
            }
        }
    }
}