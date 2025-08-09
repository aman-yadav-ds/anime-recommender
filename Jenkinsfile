pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = 'mlops-new-447207'
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
        // KUBECTL_AUTH_PLUGIN = "/usr/lib/google-cloud-sdk/bin"
    }

    stages{

        stage("Cloning from Github...."){
            steps{
                script{
                    echo 'Cloning from Github...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/aman-yadav-ds/anime-recommender.git']])
                }
            }
        }

        stage("Making a virtual environment...."){
            steps{
                script{
                    echo 'Making a virtual environment...'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .[development]
                    '''
                }
            }
        }


        stage('DVC Operations'){
            steps{
                withCredentials([file(credentialsId:'recommender-gcp-key' , variable: 'GOOGLE_APPLICATION_CREDENTIALS' )]){
                    script{
                        echo 'DVC Operations....'
                        sh """
                        . ${VENV_DIR}/bin/activate
                        dvc pull --force
                        dvc repro
                        """
                    }
                }
            }
        }


        stage('Build and Push Image to GCR'){
            steps{
                withCredentials([file(credentialsId:'recommender-gcp-key' , variable: 'GOOGLE_APPLICATION_CREDENTIALS' )]){
                    script{
                        echo 'Build and Push Image to GCR'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet
                        docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .
                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest
                        '''
                    }
                }
            }
        }


        // stage('Deploying to Kubernetes'){
        //     steps{
        //         withCredentials([file(credentialsId:'recommender-gcp-key' , variable: 'GOOGLE_APPLICATION_CREDENTIALS' )]){
        //             script{
        //                 echo 'Deploying to Kubernetes'
        //                 sh '''
        //                 export PATH=$PATH:${GCLOUD_PATH}:${KUBECTL_AUTH_PLUGIN}
        //                 gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
        //                 gcloud config set project ${GCP_PROJECT}
        //                 gcloud container clusters get-credentials ml-app-cluster --region asia-south1-a
        //                 kubectl apply -f deployment.yaml
        //                 '''
        //             }
        //         }
        //     }
        // }
    }
}