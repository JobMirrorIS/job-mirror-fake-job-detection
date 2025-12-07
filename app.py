import streamlit as st
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.sparse import hstack 

import os
import requests
from datetime import datetime

# ------------------------------------------------------------------
# Supabase configuration  (TEMP: hard-coded for local development)
# ------------------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"] 
SUPABASE_TABLE = "job_predictions" # <-- your table name


# --- PATCH for scikit-learn _RemainderColsList compatibility ---
import sklearn.compose._column_transformer as ct

if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Fallback class so older ColumnTransformer pickles can be loaded."""
        pass

    ct._RemainderColsList = _RemainderColsList
# ----------------------------------------------------------------


# PAGE CONFIGURATION (LOOK & FEEL)

st.set_page_config(
    page_title="Job Mirror - Fake Job Detection",
    page_icon="🪞",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 0.95rem;
        text-align: center;
        color: #666666;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .form-card {
        background-color: #f8f9fb;
        padding: 1.2rem 1.4rem;
        border-radius: 0.8rem;
        border: 1px solid #e2e6f0;
        margin-bottom: 1rem;
    }
    .result-header {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .footer-text {
        font-size: 0.8rem;
        color: #999999;
        text-align: center;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# WELCOME MESSAGE

st.markdown(
    '<div class="main-title">🪞 Job Mirror – Fake Job Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Enter the details of a job posting below. '
    'Job Mirror will predict whether the posting is likely '
    '<b>FAKE</b> or <b>REAL</b> and highlight the key reasons behind the decision.</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD MODEL & PREPROCESSORS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb = XGBClassifier()
    xgb.load_model("xgb_model.json")   # Load model JSON

    prep = joblib.load("preprocessing_artifacts_only.joblib")  # Load preprocessing objects

    return {
        "model": xgb,
        "tfidf_desc": prep["tfidf_desc"],
        "tfidf_req": prep["tfidf_req"],
        "preprocessor_other": prep["preprocessor_other"],
        "salary_scaler": prep["salary_scaler"],
        "ohe_salary_start": prep["ohe_salary_start"],
        "ohe_salary_end": prep["ohe_salary_end"],
        "bins": prep["bins"],
        "bin_labels": prep["bin_labels"],
    }

artifacts = load_artifacts()
xgb_model = artifacts["model"]
tfidf_desc = artifacts["tfidf_desc"]
tfidf_req = artifacts["tfidf_req"]
preprocessor_other = artifacts["preprocessor_other"]
salary_scaler = artifacts["salary_scaler"]
ohe_salary_start = artifacts["ohe_salary_start"]
ohe_salary_end = artifacts["ohe_salary_end"]
bins = artifacts["bins"]
bin_labels = artifacts["bin_labels"]

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def parse_salary_range(s):
    if s is None or s == "" or s == "unknown":
        return np.nan, np.nan
    if "-" not in s:
        return np.nan, np.nan
    try:
        v = s.split("-")
        return int(v[0]), int(v[1])
    except:
        return np.nan, np.nan


def build_features_from_input(job):
    df = pd.DataFrame([job])

    # Salary
    df["salary_range"] = df["salary_range"].fillna("unknown")
    df[["start_salary", "end_salary"]] = df["salary_range"].apply(
        lambda x: pd.Series(parse_salary_range(x))
    )
    df["start_salary"] = df["start_salary"].fillna(0)
    df["end_salary"] = df["end_salary"].fillna(0)
    df["start_salary"] = np.log1p(df["start_salary"])
    df["end_salary"] = np.log1p(df["end_salary"])
    df[["start_salary_scaled", "end_salary_scaled"]] = salary_scaler.transform(
        df[["start_salary", "end_salary"]]
    )
    df["salary_bin_start"] = pd.cut(
        df["start_salary_scaled"], bins=bins, labels=bin_labels, include_lowest=True, right=False
    )
    df["salary_bin_end"] = pd.cut(
        df["end_salary_scaled"], bins=bins, labels=bin_labels, include_lowest=True, right=False
    )
    Xss = ohe_salary_start.transform(df[["salary_bin_start"]])
    Xse = ohe_salary_end.transform(df[["salary_bin_end"]])

    # Text fields
    for col in ["description", "requirements", "benefits", "company_profile"]:
        df[col] = df[col].fillna("")

    X_desc = tfidf_desc.transform(df["description"])
    X_req = tfidf_req.transform(df["requirements"])

    # Other fields
    other_cols = [
        "location", "department", "telecommuting", "has_company_logo", "has_questions",
        "employment_type", "required_experience", "required_education", "industry", "function"
    ]

    for col in other_cols:
        if col not in df:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown")

    X_other = preprocessor_other.transform(df[other_cols])

    # Final feature stack
    return hstack([X_desc, X_req, X_other, Xss, Xse])


def explain(job):
    reasons = []

    if job.get("salary_range", "unknown") == "unknown":
        reasons.append("Salary information is missing or unknown.")

    if int(job.get("has_company_logo", 1)) == 0:
        reasons.append("Company logo is missing.")

    if int(job.get("telecommuting", 0)) == 1:
        reasons.append("Job is fully remote.")

    if not job.get("company_profile", "").strip():
        reasons.append("Company profile section is empty.")

    if not job.get("benefits", "").strip():
        reasons.append("Benefits section is empty.")

    if len(job.get("description", "")) < 100:
        reasons.append("Job description is unusually short.")

    suspicious = ["earn", "money", "weekly", "work from home", "no experience"]
    text = job.get("description", "").lower()
    for w in suspicious:
        if w in text:
            reasons.append(f"Suspicious keyword detected: '{w}'.")

    return reasons


def predict_job(job):
    X = build_features_from_input(job)
    prob_fake = float(xgb_model.predict_proba(X)[0, 1])
    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    return {
        "label": label,
        "prob_fake": prob_fake,
        "prob_real": 1 - prob_fake,
        "reasons": explain(job)
    }

# ------------------------------------------------------------------
# Helper: Save prediction + input to Supabase
# ------------------------------------------------------------------
def save_prediction_to_supabase(job_input: dict, result: dict):
    """
    Sends one row to the 'job_predictions' table in Supabase.
    job_input: your job_dict (all user inputs)
    result:    dict from predict_job() containing label, prob_fake, prob_real, reasons
    """
    # Build REST endpoint for the table
    url = f"{SUPABASE_URL}/rest/v1/job_predictions"

    # Supabase expects the anon key both as apikey and Bearer token
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",  # ask Supabase to return the row we inserted
    }

    # Convert reasons list → single string
    reasons = result.get("reasons", [])
    if isinstance(reasons, list):
        reasons_text = "; ".join(reasons)
    else:
        reasons_text = str(reasons)

    # Map your input + output to the Supabase table columns
    payload = {
        # --- text fields from the form ---
        "description":        job_input.get("description", ""),
        "requirements":       job_input.get("requirements", ""),
        "benefits":           job_input.get("benefits", ""),
        "company_profile":    job_input.get("company_profile", ""),
        "salary_range":       job_input.get("salary_range", ""),
        "location":           job_input.get("location", ""),
        "department":         job_input.get("department", ""),
        "industry":           job_input.get("industry", ""),
        "function":           job_input.get("function", ""),
        "employment_type":    job_input.get("employment_type", ""),
        "required_experience": job_input.get("required_experience", ""),
        "required_education":  job_input.get("required_education", ""),

        # --- boolean flags ---
        "telecommuting":    bool(job_input.get("telecommuting", 0)),
        "has_company_logo": bool(job_input.get("has_company_logo", 0)),
        "has_questions":    bool(job_input.get("has_questions", 0)),

        # --- model outputs ---
        "label":     result.get("label", ""),
        "prob_fake": float(result.get("prob_fake", 0.0)),
        "prob_real": float(result.get("prob_real", 0.0)),
        "reasons":   reasons_text,
        # created_at will be filled automatically by default now() in Supabase
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code not in (200, 201):
            # Log error to Streamlit console, but don't break the app
            st.warning(f"Could not log prediction to Supabase (status {resp.status_code}).")
    except Exception as e:
        st.warning(f"Error saving prediction to Supabase: {e}")


# ---------------------------------------------------------
# STREAMLIT INPUT FORM
# ---------------------------------------------------------
st.markdown(
    '<div class="section-header">✏️ Enter Job Posting Details</div>', 
    unsafe_allow_html=True
    )

# 🔹 Wrap the entire form in a soft card container
with st.form("job_form"):
        col1, col2 = st.columns(2)

        with col1:
            description = st.text_area(
                "Job Description *",
                height=180,
                placeholder="Paste the full job description here..."
            )
            requirements = st.text_area(
                "Requirements",
                height=120,
                placeholder="Mention key skills, experience, and tools..."
            )
            benefits = st.text_area(
                "Benefits",
                height=100,
                placeholder="Optional: salary perks, health benefits, etc."
            )
            company_profile = st.text_area(
                "Company Profile",
                height=100,
                placeholder="Optional: brief company background..."
            )

        with col2:
            salary_range = st.text_input("Salary Range", value="unknown", help="Example: 50000-80000 or unknown")
            location = st.text_input("Location", value="US, CA, San Francisco")
            department = st.text_input("Department", value="Data & Analytics")

            st.markdown("---")
            telecommuting = st.checkbox("Remote / Telecommuting")
            has_company_logo = st.checkbox("Has Company Logo", value=True)
            has_questions = st.checkbox("Has Screening Questions", value=True)

            employment_type = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"],
                index=0
            )
            required_experience = st.selectbox(
                "Required Experience",
                ["Not Applicable", "Entry level", "Mid-Senior level", "Director", "Executive"],
                index=2
            )
            required_education = st.selectbox(
                "Required Education",
                ["Unspecified", "High School or equivalent", "Associate's Degree", "Bachelor's Degree",
                 "Master's Degree", "Doctorate"],
                index=3
            )

            industry = st.text_input("Industry", value="Information Technology")
            function = st.text_input("Function", value="Analyst")

            submitted = st.form_submit_button("🔍 Predict Fraud Risk")
    
st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# PREDICTION OUTPUT SECTION
# ------------------------------------------------------------
if submitted:
    job_dict = {
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "company_profile": company_profile,
        "salary_range": salary_range,
        "location": location,
        "department": department,
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
    }

    result = predict_job(job_dict)

    #  NEW: Log prediction + inputs to Supabase
    save_prediction_to_supabase(job_dict, result)

    # UI Rendering starts here
    st.markdown("---")
    st.markdown(
    '<div class="section-header">📊 Prediction Result</div>',
    unsafe_allow_html=True,
)

    # Two columns: left = label + reason, right = metrics
    left_col, right_col = st.columns([2, 1])

    with left_col:
       if result["label"] == "FAKE":
         st.error(
            f"⚠️ This job posting appears **FAKE**.\n\n"
            f"**Probability Fake:** {result['prob_fake'] * 100:.1f}%\n\n"
            f"**Probability Real:** {result['prob_real'] * 100:.1f}%"
        )

       if result["reasons"]:
            st.markdown("### ‼️ This job seems fake because:")
            for r in result["reasons"]:
                st.markdown(f"- {r}")
       else:
        st.success(
            f"✅ This job posting appears **REAL**.\n\n"
            f"**Probability Real:** {result['prob_real'] * 100:.1f}%\n\n"
            f"**Probability Fake:** {result['prob_fake'] * 100:.1f}%"
        )

        if result["reasons"]:
            st.markdown("### ⚠️ But note the following:")
            for r in result["reasons"]:
                st.markdown(f"- {r}")

    with right_col:
     st.metric("Probability of REAL", f"{result['prob_real'] * 100:.1f}%")
     st.metric("Probability of FAKE", f"{result['prob_fake'] * 100:.1f}%")
