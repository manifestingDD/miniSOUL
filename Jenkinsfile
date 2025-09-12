pipeline {
    agent any

    environment {
        GCP_PROJECT_ID      = 'minisoul'
        IMAGE_NAME          = 'minisoul'
        REGION              = 'us-central1'
        ARTIFACT_REGISTRY   = "${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/minisoul-repo"
        IMAGE_TAG           = "latest"
        IMAGE_URI           = "${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        CLOUD_RUN_SERVICE   = 'minisoul-app'
    }

    stages {
        stage('1. Checkout from GitHub') {
            steps {
                git branch: 'main', url: 'https://github.com/manifestingDD/miniSOUL.git'
                echo "✅ Code checked out successfully."
            }
        }

        stage('2. Build Docker Image') {
            steps {
                script {
                    sh "docker build -t ${IMAGE_URI} ."
                    echo "✅ Docker image built and tagged: ${IMAGE_URI}"
                }
            }
        }

        stage('3. Push to GCP Artifact Registry') {
            steps {
                withCredentials([string(credentialsId: 'gcp-service-account-json', variable: 'GCP_KEY')]) {
                    script {
                        // Write the key to a temporary file
                        sh '''
                            echo $GCP_KEY > /tmp/gcp-key.json
                            gcloud auth activate-service-account --key-file=/tmp/gcp-key.json
                            gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
                            docker push ${IMAGE_URI}
                            rm /tmp/gcp-key.json
                        '''
                        echo "✅ Image pushed successfully to Artifact Registry."
                    }
                }
            }
        }

        stage('4. Deploy to GCP Cloud Run') {
            steps {
                withCredentials([string(credentialsId: 'gcp-service-account-json', variable: 'GCP_KEY')]) {
                    script {
                        sh '''
                            echo $GCP_KEY > /tmp/gcp-key.json
                            gcloud auth activate-service-account --key-file=/tmp/gcp-key.json
                            gcloud run deploy ${CLOUD_RUN_SERVICE} \
                              --platform managed \
                              --allow-unauthenticated \
                              --region ${REGION} \
                              --image ${IMAGE_URI} \
                              --quiet
                            rm /tmp/gcp-key.json
                        '''
                        echo "🚀 App successfully deployed to Cloud Run!"
                    }
                }
            }
        }
    }
}