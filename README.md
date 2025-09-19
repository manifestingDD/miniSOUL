# miniSOUL: MMM Scenario Planning & Optimization Lab

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Build-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI/CD-Jenkins%20%7C%20GCP-orange.svg)](https://www.jenkins.io/)

**Live Demo:** [**miniSOUL Application**](http://YOUR_CLOUD_RUN_URL_HERE) 🚀

## 📖 Introduction

**SOUL (Simulation and Optimization Unified Lab)** is an in-house interactive web application built with Streamlit for Marketing Mix Model (MMM) scenario planning. It empowers users to move beyond historical analysis and proactively plan for the future by simulating outcomes based on different budget allocations and optimization given planner's specified target and constaints. The tool is equipped with a customized greedy optimization algorithm to help users make data-driven decisions.

This MiniSoul project is built to demonstrate the app functionalities and showcasing modern MLOps practices in terms of a full, end-to-end CI/CD pipeline for a data science application.

## ✨ Key Features

This multi-page application provides three core functionalities:

* **Scenario Planning:** User provide manually adjust monthly media spend as CSV files and instantly see the forecasted impact on key performance indicators (KPIs), such as attendance or revenue.
* **Budget Optimizing:** User set a target KPI goal, and the application's optimization engine will find the minimum budget required to achieve it, reallocating spend across channels and months.
* **Attendee Maximizing:** User provide an initial budget plan, and the optimizer will reallocate spending to achieve the maximum possible number of attendees or other target KPIs using the same total budget as the initial plan.

## ✨ Optimization Algorithm
The MMM set up and media planning process in practice brings 3 major challenges for planning and optimizing:
* **Non-linear mapping function:** The adstock and saturation effect of media spend means complex non-linear mapping from week 1's spend on media channel $X$ to incremental return on week 7, bringing challenge to the optimizer. 
* **Interdependent choice variables:** As the saturation effect is applied to cumulative media adstock, the return of week 3's investment is dependent on week 1 and week 2's investment.  
* **Unmatched granularity at modeling and planning stage:** Stakeholders might ask for more granualar ROI information when fitting the MMM model but more high level recommendation when making the actual plan. For example, a channel could be separated into 3 subchannels by touchpoint phase {Inspire, Activate, Transact} when fitting the MMM, but the 3 sub-channels will be aggregated together during planning. This brings the challenge for an arithmatic optimization solution as we lack the adstock and statuarion parameters for the aggregated channel. 


## 🛠️ Tech Stack & CI/CD Architecture

This project utilizes a modern stack for both the application and its deployment infrastructure.

| Category      | Technology                                    | Purpose                                                 |
| ------------- | --------------------------------------------- | ------------------------------------------------------- |
| **Frontend**  | Streamlit                                     | Building the interactive multi-page web application.    |
| **Backend**   | Python, Pandas, NumPy                         | Data manipulation and optimization algorithms.          |
| **Container** | Docker                                        | Packaging the app and its dependencies for deployment.  |
| **CI/CD**     | Jenkins                                       | Automation server for the CI/CD pipeline.               |
| **Cloud**     | Google Cloud Platform (GCP)                   | Hosting and infrastructure.                             |
| **- Compute** | Compute Engine                                | Hosting the Jenkins server as a VM.                     |
| **- Registry**| Artifact Registry                             | Storing and managing the built Docker images.           |
| **- Serving** | Cloud Run                                     | Serverless platform for deploying and running the app.  |
| **- Logging** | Cloud Logging                                 | Centralized logging for the deployed application.       |

### Pipeline Architecture

The CI/CD pipeline is fully automated. A `git push` to the `main` branch on GitHub triggers a webhook that starts a build job on a Jenkins server hosted on a GCP Compute Engine VM.

+------------------------+
|   Author Pushes Code   |
+------------------------+
           |
           v
+------------------------+
|         GitHub         |
+------------------------+
           |
           | (Webhook Trigger)
           v
+------------------------+
|   Jenkins (GCP VM)     |
+------------------------+
           |
           | (Builds Container)
           v
+------------------------+
|         Docker         |
+------------------------+
           |
           | (Pushes Image)
           v
+------------------------+
| GCP Artifact Registry  |
+------------------------+
           |
           | (Deploys New Version)
           v
+------------------------+
|     GCP Cloud Run      |
+------------------------+
           |
           v
+------------------------+
|  User Accesses App     |
+------------------------+


The Jenkins pipeline executes the following stages:
1.  **Checkout:** Pulls the latest code from the GitHub repository.
2.  **Build Docker Image:** Builds a new Docker image based on the `Dockerfile`.
3.  **Push to GCP Artifact Registry:** Tags the new image and pushes it to a private Artifact Registry repository.
4.  **Deploy to GCP Cloud Run:** Deploys the new image to the Cloud Run service, making the update live.


## ⚙️ Local Setup and Installation

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/manifestingDD/miniSOUL.git](https://github.com/manifestingDD/miniSOUL.git)
    cd miniSOUL
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.


