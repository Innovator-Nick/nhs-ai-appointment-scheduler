# NHS AI Appointment Scheduler

An artificial intelligence system that predicts GP appointment no-shows using machine learning and displays risk insights through an interactive dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Purpose](#purpose-of-this-project)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Machine Learning Model Training](#machine-learning-model-training)
- [Dataset Information](#dataset-information)
- [System Architecture](#system-architecture)
- [Screenshots](#screenshots)
- [How to Run the Project](#how-to-run-the-project)
- [Future Improvements](#future-improvements)
- [Author & Contact](#author-&-contact)
- [License](#license)

---

## Overview

The NHS AI Appointment Scheduler is a machine learning–powered system designed to predict patient appointment no-shows (DNA – Did Not Attend).

Missed appointments create operational inefficiencies and increase healthcare costs. This system uses artificial intelligence to predict no-show risk and visualise results in an interactive dashboard.

This project demonstrates the real-world application of AI in healthcare scheduling and predictive analytics.

---

## Purpose of This Project

This project was developed to demonstrate the use of artificial intelligence in predicting healthcare appointment attendance behaviour and helping reduce missed appointments.

It showcases end-to-end AI development, including:

- Model training
- Backend API integration
- Frontend dashboard
- Real-time predictions

---

## Key Features

- AI model trained to predict appointment no-shows
- 93.40% prediction accuracy
- Interactive dashboard with charts
- Real-time prediction tool
- Risk score visualisation
- Backend API integration
- Fully working localhost deployment

---

## Technology Stack

### Backend

- Python
- FastAPI / Flask
- Scikit-learn
- Pandas
- NumPy

### Frontend

- HTML
- CSS
- JavaScript

### Tools

- Localhost server
- REST API

---

## Machine Learning Model Training

The machine learning model is automatically trained when the backend server starts.

Training is implemented in:

backend/app/main.py


The system performs:

- Dataset loading
- Model training
- Accuracy evaluation
- Prediction system initialisation

Example terminal output:




This confirms the AI model is fully operational.

---

## Dataset Information

This system was trained and tested using synthetic data.

Synthetic data was used for research and demonstration purposes.

No real patient data was used.

---

## System Architecture

The system consists of two main components:

### Backend

- Python-based API
- Machine learning model
- Prediction engine

### Frontend

- Interactive dashboard
- Visual charts and analytics
- Prediction interface

Frontend communicates with backend via API requests.

---

## Screenshots

### Dashboard and Prediction

![Dashboard and Prediction](screenshots/dashboard-prediction.png)


### DNA Charts

![DNA Charts](screenshots/dna-charts.png)


### Model Training Output

![Terminal](screenshots/terminal.png)


---

## How to Run the Project

### Step 1 — Start Backend

Open terminal:

cd backend
python app/main.py

Expected output:

Model trained! Accuracy: 93.40%
NHS AI Scheduler API started successfully!

---

### Step 2 — Start Frontend

Open browser:

http://localhost:3000


Dashboard will load.

---

## Future Improvements

- Deploy system to cloud environment
- Integrate with real healthcare datasets (subject to approval)
- Improve model performance
- Production-level deployment
- Integration with healthcare providers

---

## Author & Contact

**Innovator-Nick**

MSc Artificial Intelligence, 2024

Independent AI Developer

Solo Founder


---

## License

This project is licensed under the MIT License.

This project is provided for research and educational purposes.

---

