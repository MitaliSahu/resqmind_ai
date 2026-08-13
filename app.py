import streamlit as st
import pandas as pd
from database import init_db, add_patient, get_queue, get_resources
from ml_engine import predict_triage
from rag_assistant import ask_rag_assistant

# Initialize Database on launch
init_db()

st.set_page_config(page_title="ResQMind AI", page_icon="🏥", layout="wide")

st.title("🏥 ResQMind AI — Emergency Decision Support Platform")
st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📋 Patient Intake & Triage", "👨‍⚕️ Hospital Queue & Resources", "🤖 RAG Clinical Copilot"])

# TAB 1: Patient Registration & Machine Learning Triage
with tab1:
    st.header("Patient Triage Entry")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Patient Name", "John Doe")
        age = st.number_input("Age", min_value=1, max_value=110, value=45)
        symptoms = st.text_area("Symptoms", "Chest pain, shortness of breath")
        med_history = st.text_input("Medical History", "Hypertension")

    with col2:
        hr = st.number_input("Heart Rate (BPM)", min_value=30, max_value=220, value=110)
        spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=92)
        temp = st.number_input("Body Temp (°F)", min_value=94.0, max_value=108.0, value=99.5)

    if st.button("Run AI Triage & Register"):
        # Predict using Machine Learning Model
        priority, severity = predict_triage(age, hr, spo2, temp)
        vitals_str = f"HR: {hr} | SpO2: {spo2}% | Temp: {temp}°F"
        
        # Save to SQLite Database
        add_patient(name, age, symptoms, vitals_str, med_history, priority, severity)
        
        st.success(f"Patient registered! Priority Assigned: **{priority}** (Severity Score: {severity})")

# TAB 2: Real-time Priority Queue & Resource Tracker
with tab2:
    st.header("Hospital Command Center")
    
    # Display Hospital Resources
    res_data = get_resources()
    res_cols = st.columns(len(res_data))
    for idx, item in enumerate(res_data):
        res_cols[idx].metric(label=item[0], value=f"{item[1]} / {item[2]}")
        
    st.subheader("Prioritized Patient Queue")
    patients = get_queue()
    if patients:
        df = pd.DataFrame(patients, columns=["ID", "Name", "Age", "Symptoms", "Vitals", "History", "Priority", "Severity Score", "Arrival Time"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No patients currently in queue.")

# TAB 3: RAG Medical Assistant
with tab3:
    st.header("Clinical Guidelines RAG Copilot")
    query = st.text_input("Ask a clinical question or protocol justification:", "What is the emergency protocol for SpO2 below 88%?")
    if st.button("Ask Assistant"):
        with st.spinner("Searching medical guidelines..."):
            answer = ask_rag_assistant(query)
            st.write(answer)