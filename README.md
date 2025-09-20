# miniSOUL: MMM Scenario Planning & Optimization Lab

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Build-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI/CD-Jenkins%20%7C%20GCP-orange.svg)](https://www.jenkins.io/)

**Live Demo:** [**miniSOUL Application**](https://minisoul-app-354747708193.us-central1.run.app/) 🚀

## 📖 Introduction

**SOUL (Simulation and Optimization Unified Lab)** is an in-house interactive web application built with Streamlit for Marketing Mix Model (MMM) scenario planning. It empowers users to move beyond historical analysis and proactively plan for the future by simulating outcomes based on different budget allocations and optimization given planner's specified target and constaints. The tool is equipped with a customized greedy optimization algorithm to help users make data-driven decisions.

This MiniSoul project is built to demonstrate the app functionalities and showcasing modern MLOps practices in terms of a full, end-to-end CI/CD pipeline for a data science application.

## ✨ Key Features

This multi-page application provides three core functionalities:

* **Scenario Planning:** User provide manually adjust monthly media spend as CSV files and instantly see the forecasted impact on key performance indicators (KPIs), such as attendance or revenue.
* **Budget Optimizing:** User set a target KPI goal, and the application's optimization engine will find the minimum budget required to achieve it, reallocating spend across channels and months.
* **Attendee Maximizing:** User provide an initial budget plan, and the optimizer will reallocate spending to achieve the maximum possible number of attendees or other target KPIs using the same total budget as the initial plan.

## 🗻  Optimization Algorithm

The MMM set up and media planning process in practice brings 3 major challenges for planning and optimizing:
* **Non-linear mapping function:** The adstock and saturation effect of media spend means complex non-linear mapping from week 1's spend on media channel $X$ to incremental return on week 7, bringing challenge to the optimizer. 
* **Interdependent choice variables:** As the saturation effect is applied to cumulative media adstock, the return of week 3's investment is dependent on week 1 and week 2's investment.  
* **Unmatched granularity at modeling and planning stage:** Stakeholders might ask for more granualar ROI information when fitting the MMM model but more high level recommendation when making the actual plan. For example, a channel could be separated into 3 subchannels by touchpoint phase {Inspire, Activate, Transact} when fitting the MMM, but the 3 sub-channels will be aggregated together during planning. This brings the challenge for an arithmatic optimization solution as we lack the adstock and statuarion parameters for the aggregated channel. 

To address these challenges, we developed an in-house optimization algorithm that combines exhaustive simulation with an efficient greedy search. The major steps are: 
* **1.Pre-computation via Simulation:**  For each media channel, we first simulate 300 different spending levels (from 0% to 300% of modeling spend as baseline). At each level, we calculate the resulting incremental return and, most importantly, the Marginal Cost Per Incremental Target (MCPT). This process creates a detailed, 3000-row lookup table for each channel that acts as a cost-benefit curve, showing us exactly how the efficiency of a channel changes as we invest more into it.
* **2. Handling Adstock & Timing:** We recognize that a dollar spent in March has a different ROI than a dollar spent in January for a Q1 target period. We capture this by applying an "estimated timing vector" (derived from the model's adstock parameters) to our simulation table. This effectively treats "Media X in January" and "Media X in February" as distinct investment options with their own unique cost-benefit curves.
* **3. The Greedy Algorithm:** With the simulation complete, the optimizer iteratively allocates the budget one small piece at a time. In each step, it asks a simple question: "Where can I spend the next dollar to get the best possible return?" It finds the media channel and month combination with the lowest current MCPT, allocates a small amount of budget there untill reaching the media with 2nd lowest MCPT, and update the MCPT tracker. This process repeats until the budget is exhausted or the goal is met. 

This simulation-first approach allows the greedy algorithm to navigate the complex, non-linear search space effectively and find a near-optimal solution quickly. The key proposition for this method is that by general assumptions of media mix model, the impact of MMM estimates are locally linear -- media channels that has over +/- 50% spending changes compared to the modeling period are encouraged to be re-estimated due to the media saturation effect. This proposition gurantees the finite scenarios that need to be simulated. Although we simulate the spend scenario from 0% to 300% of original spend, we encourage the users to set the constraint as (50%, 150%) for each media channel. 



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

[Author Pushes Code] --> [GitHub] --(Webhook)--> [Jenkins on GCP VM] --(Builds)--> [Docker Image] --(Pushes)--> [Artifact Registry] --(Deploys)--> [Cloud Run] <-- [User Accesses App]


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


