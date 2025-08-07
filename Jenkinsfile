pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = 'anime-recommender-system'
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
        KUBECTL_AUTH_PLUGIN = "/usr/lib/google-cloud-sdk/bin"
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
        stage("Build and push Image to GCR....") {
            steps {
                withCredentials([file(credentialsId:'recommender-gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script {
                        echo "Building app Image and pushing it to GCR...."
                        sh """
                        . ${VENV_DIR}/bin/activate
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet
                        docker build -t gcr.io/${GCP_PROJECT}/anime-recommender:latest .
                        docker push gcr.io/${GCP_PROJECT}/anime-recommender:latest
                        """
                    }
                }
            }
        }
    }
}