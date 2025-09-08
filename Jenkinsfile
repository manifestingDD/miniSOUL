// Jenkinsfile
pipeline {
    agent any // This tells Jenkins to grab any available agent/server to run the pipeline.

    // Environment variables that will be used throughout the pipeline.
    // We will set up the credentials in Jenkins later, not hardcode them here.
    environment {
        GCP_PROJECT_ID      = 'the-gcp-project-id' // Placeholder
        GCP_CREDENTIALS     = credentials('gcp-service-account-key') // ID of credentials in Jenkins
        IMAGE_NAME          = 'minisoul'
        REGION              = 'us-central1' // e.g., us-central1
        ARTIFACT_REGISTRY   = "${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/minisoul-repo"
        IMAGE_TAG           = "latest"
        IMAGE_URI           = "${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        CLOUD_RUN_SERVICE   = 'minisoul-app'
    }

    stages {
        stage('1. Checkout from GitHub') {
            steps {
                // This command checks out the code from the GitHub repository.
                // Jenkins automatically knows the repository URL from the pipeline setup.
                git branch: 'main', url: 'https://github.com/manifestingDD/miniSOUL.git'
                echo "✅ Code checked out successfully."
            }
        }

        stage('2. Build Docker Image') {
            steps {
                script {
                    // This command builds the Docker image using the Dockerfile in the repo.
                    // It tags the image with the URI for the future GCP Artifact Registry.
                    sh "docker build -t ${IMAGE_URI} ."
                    echo "✅ Docker image built and tagged: ${IMAGE_URI}"
                }
            }
        }

        stage('3. Push to GCP Artifact Registry') {
            steps {
                script {
                    // We use the gcloud CLI (which we will install on the Jenkins server)
                    // to authenticate Docker with our GCP Artifact Registry.
                    sh "gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet"
                    
                    // Push the tagged Docker image to the registry.
                    sh "docker push ${IMAGE_URI}"
                    echo "✅ Image pushed successfully to Artifact Registry."
                }
            }
        }

        stage('4. Deploy to GCP Cloud Run') {
            steps {
                script {
                    // This gcloud command deploys the new image to Cloud Run.
                    // --platform 'managed' specifies using the serverless platform.
                    // --allow-unauthenticated allows public access to the app.
                    // --region specifies the deployment region.
                    // --image points to the image we just pushed.
                    sh """
                    gcloud run deploy ${CLOUD_RUN_SERVICE} \
                      --platform managed \
                      --allow-unauthenticated \
                      --region ${REGION} \
                      --image ${IMAGE_URI} \
                      --quiet
                    """
                    echo "🚀 App successfully deployed to Cloud Run!"
                }
            }
        }
    }
}