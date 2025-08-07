pipeline {
    agent any

    stages {
        stage("Cloning from Github....") {
            steps {
                script {
                    echo 'Cloning from Github.....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/aman-yadav-ds/anime-recommender.git']])
                }
            }
        }
    }
}