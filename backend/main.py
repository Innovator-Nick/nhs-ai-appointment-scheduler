from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json
from typing import List, Dict
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Healthacre AI Appointment Scheduler",
    description="AI-powered appointment scheduling for NHS practices",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
demo_data = None
feature_names = None


def generate_demo_data():
    """Generate realistic NHS appointment data"""
    np.random.seed(42)
    n_appointments = 2000

    # Generate realistic data
    ages = np.random.normal(45, 20, n_appointments).clip(0, 100)
    genders = np.random.choice(['M', 'F'], n_appointments)
    imd_deciles = np.random.choice(range(1, 11), n_appointments)

    appointment_types = np.random.choice([
        'GP Consultation', 'Nurse Consultation', 'Mental Health',
        'Chronic Disease Review', 'Vaccination', 'Health Check'
    ], n_appointments, p=[0.4, 0.25, 0.1, 0.1, 0.08, 0.07])

    hours = np.random.choice(range(8, 18), n_appointments)
    days_of_week = np.random.choice(range(5), n_appointments)
    lead_times = np.random.exponential(7, n_appointments).clip(0, 60)
    previous_dna_count = np.random.poisson(0.5, n_appointments)

    # Generate DNA outcomes
    dna_probabilities = (
            0.05 +
            (ages < 30) * 0.03 +
            (imd_deciles > 7) * 0.04 +
            (lead_times > 14) * 0.02 +
            (hours < 10) * 0.02
    )
    dna_outcomes = np.random.binomial(1, dna_probabilities, n_appointments)

    return pd.DataFrame({
        'patient_id': range(1, n_appointments + 1),
        'age': ages.astype(int),
        'gender': genders,
        'imd_decile': imd_deciles,
        'appointment_type': appointment_types,
        'hour': hours,
        'day_of_week': days_of_week,
        'booking_lead_time': lead_times.round(1),
        'previous_dna_count': previous_dna_count,
        'dna': dna_outcomes
    })


def train_ai_model():
    """Train the DNA prediction model"""
    global model, demo_data, feature_names

    demo_data = generate_demo_data()

    # Feature engineering
    demo_data['gender_encoded'] = demo_data['gender'].map({'M': 0, 'F': 1})

    # Get dummies for appointment types
    appointment_dummies = pd.get_dummies(demo_data['appointment_type'], prefix='appt')
    demo_data = pd.concat([demo_data, appointment_dummies], axis=1)

    # Define features
    feature_names = ['age', 'imd_decile', 'hour', 'day_of_week',
                     'booking_lead_time', 'previous_dna_count', 'gender_encoded']
    feature_names.extend(appointment_dummies.columns)

    # Train model
    X = demo_data[feature_names]
    y = demo_data['dna']

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    print(f"Model trained! Accuracy: {model.score(X, y):.2%}")
    print(f"Overall DNA rate: {y.mean():.2%}")


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    train_ai_model()
    print("🏥 Healthacre AI Scheduler API started successfully!")


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Healthacre AI Appointment Scheduler API", "status": "running"}


@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get data for the dashboard"""
    if demo_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Calculate key metrics
    total_appointments = len(demo_data)
    dna_count = demo_data['dna'].sum()
    dna_rate = dna_count / total_appointments

    # Monthly trends
    demo_data['month'] = pd.date_range('2024-01-01', periods=len(demo_data), freq='D').strftime('%Y-%m')
    monthly_data = demo_data.groupby('month').agg({
        'dna': ['count', 'sum', 'mean']
    }).round(3)
    monthly_data.columns = ['total_appointments', 'dna_count', 'dna_rate']
    monthly_trends = monthly_data.reset_index().to_dict('records')

    # DNA by appointment type
    type_stats = demo_data.groupby('appointment_type')['dna'].agg(['mean', 'count']).reset_index()
    type_stats.columns = ['appointment_type', 'dna_rate', 'count']
    appointment_types = type_stats.to_dict('records')

    # Hourly patterns
    hourly_stats = demo_data.groupby('hour')['dna'].mean().reset_index()
    hourly_patterns = hourly_stats.to_dict('records')

    # ROI calculations
    current_loss = dna_count * 85  # £85 per appointment
    ai_15_reduction = current_loss * 0.85
    ai_25_reduction = current_loss * 0.75

    return {
        "summary": {
            "total_appointments": total_appointments,
            "dna_count": int(dna_count),
            "dna_rate": round(dna_rate, 3),
            "potential_annual_savings": int((current_loss - ai_25_reduction) * 2)  # Extrapolate to annual
        },
        "monthly_trends": monthly_trends,
        "appointment_types": appointment_types,
        "hourly_patterns": hourly_patterns,
        "roi_scenarios": [
            {"scenario": "Current", "monthly_cost": int(current_loss / 6)},
            {"scenario": "15% AI Reduction", "monthly_cost": int(ai_15_reduction / 6)},
            {"scenario": "25% AI Reduction", "monthly_cost": int(ai_25_reduction / 6)}
        ]
    }


@app.post("/api/predict-dna")
async def predict_dna(patient_data: dict):
    """Predict DNA probability for a patient"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Create feature vector
        features = pd.DataFrame([{
            'age': patient_data.get('age', 40),
            'imd_decile': patient_data.get('imd_decile', 5),
            'hour': patient_data.get('hour', 10),
            'day_of_week': patient_data.get('day_of_week', 1),
            'booking_lead_time': patient_data.get('booking_lead_time', 7),
            'previous_dna_count': patient_data.get('previous_dna_count', 0),
            'gender_encoded': 1 if patient_data.get('gender', 'F') == 'F' else 0
        }])

        # Add appointment type dummies
        appt_type = patient_data.get('appointment_type', 'GP Consultation')
        for col in feature_names:
            if col.startswith('appt_'):
                features[col] = 1 if col == f"appt_{appt_type}" else 0

        # Ensure all features are present
        for feature in feature_names:
            if feature not in features.columns:
                features[feature] = 0

        # Predict
        probability = model.predict_proba(features[feature_names])[0][1]

        return {
            "dna_probability": round(probability, 3),
            "risk_level": "High" if probability > 0.15 else "Medium" if probability > 0.08 else "Low",
            "recommendation": "Send extra reminders" if probability > 0.15 else "Standard process"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/api/appointments")
async def get_appointments():
    """Get sample appointments with predictions"""
    if demo_data is None or model is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Get sample appointments
    sample = demo_data.sample(50).copy()

    # Add predictions
    X_sample = sample[feature_names]
    predictions = model.predict_proba(X_sample)[:, 1]
    sample['predicted_dna_risk'] = predictions
    sample['risk_level'] = ['High' if p > 0.15 else 'Medium' if p > 0.08 else 'Low' for p in predictions]

    # Convert to list of dictionaries
    appointments = sample[['patient_id', 'age', 'appointment_type', 'hour',
                           'predicted_dna_risk', 'risk_level', 'dna']].to_dict('records')

    return {"appointments": appointments}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
