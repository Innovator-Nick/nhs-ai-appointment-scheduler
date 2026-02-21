import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPatient, setSelectedPatient] = useState({
    age: 40,
    gender: 'F',
    appointment_type: 'GP Consultation',
    hour: 10,
    imd_decile: 5,
    previous_dna_count: 0,
    booking_lead_time: 7
  });
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    fetchAppointments();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/dashboard-data`);
      setDashboardData(response.data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
  };

  const fetchAppointments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/appointments`);
      setAppointments(response.data.appointments);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching appointments:', error);
      setLoading(false);
    }
  };

  const predictDNA = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict-dna`, selectedPatient);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error predicting DNA:', error);
    }
  };

  if (loading) {
    return <div className="loading">Loading Healthacre AI Dashboard...</div>;
  }

  return (
    <div className="App">
      <header className=" Healthacre-header">
        <h1>🏥 Healthacre AI Appointment Scheduler</h1>
        <p>Intelligent scheduling for better patient care</p>
      </header>

      {dashboardData && (
        <div className="dashboard">
          {/* Summary Cards */}
          <div className="summary-cards">
            <div className="card">
              <h3>Total Appointments</h3>
              <div className="metric">{dashboardData.summary.total_appointments.toLocaleString()}</div>
            </div>
            <div className="card">
              <h3>DNA Count</h3>
              <div className="metric red">{dashboardData.summary.dna_count}</div>
            </div>
            <div className="card">
              <h3>DNA Rate</h3>
              <div className="metric">{(dashboardData.summary.dna_rate * 100).toFixed(1)}%</div>
            </div>
            <div className="card">
              <h3>Potential Annual Savings</h3>
              <div className="metric green">£{dashboardData.summary.potential_annual_savings.toLocaleString()}</div>
            </div>
          </div>

          {/* Charts */}
          <div className="charts-grid">
            {/* DNA by Appointment Type */}
            <div className="chart-container">
              <h3>DNA Rate by Appointment Type</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dashboardData.appointment_types}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="appointment_type" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  <Bar dataKey="dna_rate" fill="#005EB8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Hourly Patterns */}
            <div className="chart-container">
              <h3>DNA Patterns by Hour</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dashboardData.hourly_patterns}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  <Line type="monotone" dataKey="dna" stroke="#FF6B35" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* ROI Scenarios */}
            <div className="chart-container">
              <h3>Monthly Cost Scenarios</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dashboardData.roi_scenarios}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="scenario" />
                  <YAxis />
                  <Tooltip formatter={(value) => `£${value.toLocaleString()}`} />
                  <Bar dataKey="monthly_cost" fill="#00A499" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* DNA Prediction Tool */}
          <div className="prediction-tool">
            <h3>🎯 DNA Risk Prediction Tool</h3>
            <div className="prediction-form">
              <div className="form-grid">
                <div>
                  <label>Age:</label>
                  <input 
                    type="number" 
                    value={selectedPatient.age}
                    onChange={(e) => setSelectedPatient({...selectedPatient, age: parseInt(e.target.value)})}
                  />
                </div>
                <div>
                  <label>Gender:</label>
                  <select 
                    value={selectedPatient.gender}
                    onChange={(e) => setSelectedPatient({...selectedPatient, gender: e.target.value})}
                  >
                    <option value="F">Female</option>
                    <option value="M">Male</option>
                  </select>
                </div>
                <div>
                  <label>Appointment Type:</label>
                  <select 
                    value={selectedPatient.appointment_type}
                    onChange={(e) => setSelectedPatient({...selectedPatient, appointment_type: e.target.value})}
                  >
                    <option value="GP Consultation">GP Consultation</option>
                    <option value="Nurse Consultation">Nurse Consultation</option>
                    <option value="Mental Health">Mental Health</option>
                    <option value="Health Check">Health Check</option>
                  </select>
                </div>
                <div>
                  <label>Hour:</label>
                  <input 
                    type="number" 
                    min="8" 
                    max="17"
                    value={selectedPatient.hour}
                    onChange={(e) => setSelectedPatient({...selectedPatient, hour: parseInt(e.target.value)})}
                  />
                </div>
                <div>
                  <label>Previous DNAs:</label>
                  <input 
                    type="number" 
                    min="0"
                    value={selectedPatient.previous_dna_count}
                    onChange={(e) => setSelectedPatient({...selectedPatient, previous_dna_count: parseInt(e.target.value)})}
                  />
                </div>
              </div>
              <button onClick={predictDNA} className="predict-btn">Predict DNA Risk</button>
              
              {prediction && (
                <div className={`prediction-result ${prediction.risk_level.toLowerCase()}`}>
                  <h4>Prediction Result:</h4>
                  <p><strong>DNA Probability:</strong> {(prediction.dna_probability * 100).toFixed(1)}%</p>
                  <p><strong>Risk Level:</strong> {prediction.risk_level}</p>
                  <p><strong>Recommendation:</strong> {prediction.recommendation}</p>
                </div>
              )}
            </div>
          </div>

          {/* Appointments Table */}
          <div className="appointments-table">
            <h3>📅 Sample Appointments with AI Risk Assessment</h3>
            <table>
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Age</th>
                  <th>Appointment Type</th>
                  <th>Hour</th>
                  <th>Predicted Risk</th>
                  <th>Risk Level</th>
                  <th>Actual DNA</th>
                </tr>
              </thead>
              <tbody>
                {appointments.slice(0, 20).map((apt, index) => (
                  <tr key={index} className={apt.risk_level.toLowerCase()}>
                    <td>{apt.patient_id}</td>
                    <td>{apt.age}</td>
                    <td>{apt.appointment_type}</td>
                    <td>{apt.hour}:00</td>
                    <td>{(apt.predicted_dna_risk * 100).toFixed(1)}%</td>
                    <td>
                      <span className={`risk-badge ${apt.risk_level.toLowerCase()}`}>
                        {apt.risk_level}
                      </span>
                    </td>
                    <td>{apt.dna ? '❌ DNA' : '✅ Attended'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
