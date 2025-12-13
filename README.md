# Job Mirror – Fake Job Posting Detection System  
An end-to-end machine learning system that identifies fake job postings using NLP, XGBoost, explainability rules, and cloud-based logging through Supabase.  
The system includes a production-ready Streamlit application, model inference pipeline, preprocessing artifacts, and a secure deployment workflow.

---

##  Features
- Real-time prediction of FAKE vs REAL job postings  
- NLP preprocessing using TF-IDF vectorization  
- XGBoost fraud detection model  
- Rule-based explanation engine (identifies “red flags”)  
- Cloud logging of predictions using Supabase PostgreSQL  
- Deployed Streamlit application for public use  
- Modular design for future retraining or model upgrades  

---

##  Project Structure

Job-Mirror-Fake-Job-Detection/
│
├── app.py                            # Streamlit UI + backend logic + prediction pipeline
├── requirements.txt                  # Python dependencies
├── preprocessing_artifacts_only.joblib   # TF-IDF, encoders, scalers, preprocessors
├── xgb_model.json                    # Trained XGBoost model
└── README.md                         # Documentation

## 🔧 Installation & Setup

### 1. Clone or download the project
Download the ZIP or clone the repository: https://github.com/JobMirrorIS/job-mirror-fake-job-detection.git

cd job-mirror-fake-job-detection 


### 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # Mac/Linux 


### 3. Install dependencies
pip install -r requirements.txt 

##  Running the Application

Once dependencies are installed, run:

streamlit run app.py 


This launches the Job Mirror web UI in your browser.

You can now paste a job posting into the form and receive:
- Model prediction (FAKE or REAL)
- Probability scores
- Rule-based explanations ("red flags")


This launches the Job Mirror web UI in your browser.

You can now paste a job posting into the form and receive:
- Model prediction (FAKE or REAL)
- Probability scores
- Rule-based explanations ("red flags") 

##  Supabase Logging

Every prediction is logged into a Supabase PostgreSQL table through the Supabase REST API.

Logged fields include:
- User inputs (text + metadata)
- Model outputs (label + prob_fake + prob_real)
- Explanatory reasons
- Automatically generated UUID + timestamp

Credentials are securely stored using Streamlit Secrets:
st.secrets["SUPABASE_URL"]
st.secrets["SUPABASE_ANON_KEY"] 

##  Deployment (Streamlit Cloud)

### Steps:
1. Push project to GitHub
2. Go to streamlit.io → Deploy App
3. Select:
   - Repo: JobMirrorIS/job-mirror-fake-job-detection
   - Branch: main
   - File: app.py
4. Add Supabase credentials in Secrets 
5. Deploy



