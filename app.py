import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from difflib import get_close_matches
# DATABASE
conn = sqlite3.connect("healthcare.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    disease TEXT,
    medicines TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symptoms TEXT,
    disease TEXT,
    medicines TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
# PAGE
st.set_page_config(page_title="Healthcare AI", layout="wide")
st.title("🩺 Smart Healthcare System")
# SESSION INIT (IMPORTANT FIX)
if "user" not in st.session_state:
    st.session_state.user = None
# LOAD MODEL
model = joblib.load("naive_bayes_model.pkl")
le = joblib.load("label_encoder.pkl")
all_symptoms = joblib.load("symptoms_list.pkl")
# LOAD DATA
medicine_df = pd.read_csv("medicine.csv")
medicine_df.columns = medicine_df.columns.str.strip().str.lower()
medicine_df.rename(columns={
    "drug": "medicine",
    "medicines": "medicine"
}, inplace=True)
precaution_df = pd.read_csv("datasets/symptom_precaution.csv")
precaution_df.columns = precaution_df.columns.str.strip().str.lower()
interaction_df = pd.read_csv("datasets/db_drug_interactions.csv")
interaction_df.columns = interaction_df.columns.str.strip().str.lower()
interaction_df.rename(columns={
    "drug 1": "drug1",
    "drug 2": "drug2",
    "interaction description": "interaction"
}, inplace=True)
interaction_df["drug1"] = interaction_df["drug1"].astype(str).str.lower()
interaction_df["drug2"] = interaction_df["drug2"].astype(str).str.lower()
# FUNCTIONS
def predict(symptoms):
    vec = [0]*len(all_symptoms)
    for i, s in enumerate(all_symptoms):
        if s in symptoms:
            vec[i] = 1
    probs = model.predict_proba([vec])[0]
    idx = np.argmax(probs)
    return le.classes_[idx], probs[idx]
def get_medicine(disease):
    match = get_close_matches(disease.lower(), medicine_df["disease"].unique(), n=1, cutoff=0.5)
    if match:
        return medicine_df[medicine_df["disease"] == match[0]]
    return pd.DataFrame()
def get_precautions(disease):
    match = get_close_matches(disease.lower(), precaution_df["disease"].unique(), n=1, cutoff=0.5)
    if match:
        row = precaution_df[precaution_df["disease"] == match[0]].iloc[0]
        return [row[c] for c in precaution_df.columns if "precaution" in c and pd.notna(row[c])]
    return []
def check_interactions(predicted_meds, current_meds):
    warnings = []
    for m in predicted_meds:
        for c in current_meds:
            m, c = m.lower(), c.lower()
            match = interaction_df[
                ((interaction_df["drug1"] == m) & (interaction_df["drug2"] == c)) |
                ((interaction_df["drug1"] == c) & (interaction_df["drug2"] == m))
            ]
            if not match.empty:
                warnings.append(f"{m} ⚠️ {c} → {match.iloc[0]['interaction']}")
    return warnings
def is_new_user(user_id):
    cur.execute("SELECT * FROM feedback WHERE user_id=?", (user_id,))
    return cur.fetchone() is None
def get_user_meds(user_id):
    cur.execute("SELECT medicines FROM feedback WHERE user_id=?", (user_id,))
    res = cur.fetchone()
    if res and res[0]:
        return [x.strip().lower() for x in res[0].split(",")]
    return []
# AUTH (LOGIN / SIGNUP)
if not st.session_state.user:
    menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])
    if menu == "Signup":
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            try:
                cur.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                            (name,email,password))
                conn.commit()
                st.success("Registered Successfully")
            except:
                st.error("User already exists")
    else:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            cur.execute("SELECT * FROM users WHERE email=? AND password=?",
                        (email,password))
            user = cur.fetchone()
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")
# MAIN APP (AFTER LOGIN)
else:
    st.sidebar.success(f"👤 {st.session_state.user[1]}")
    menu = st.sidebar.selectbox("Menu", ["Home", "History", "Logout"])
    # NEW USER FEEDBACK
    if is_new_user(st.session_state.user[0]):
        st.subheader("📝 Enter Previous Health Details")
        prev_disease = st.text_input("Previous Disease")
        prev_meds = st.text_input("Medicines you are taking")

        if st.button("Save Details"):
            cur.execute("INSERT INTO feedback(user_id,disease,medicines) VALUES(?,?,?)",
                        (st.session_state.user[0], prev_disease, prev_meds))
            conn.commit()
            st.success("Saved successfully")
            st.rerun()
    # HOME
    if menu == "Home":
        symptoms = st.multiselect("Select Symptoms", all_symptoms)
        if st.button("Predict"):
            if not symptoms:
                st.warning("Select symptoms first")
            else:
                disease, conf = predict(symptoms)
                meds_df = get_medicine(disease)
                predicted_meds = []
                if not meds_df.empty and "medicine" in meds_df.columns:
                    unique_meds = meds_df["medicine"].dropna().unique()
                    filtered = [m for m in unique_meds if "/" not in str(m)]
                    top_meds = filtered[:5]
                    st.subheader("💊 Important Medicines")
                    for med in top_meds:
                        predicted_meds.append(med)
                        st.write("✔", med)
                else:
                    st.warning("No medicine data found")
                # PRECAUTIONS
                precautions = get_precautions(disease)
                if precautions:
                    st.subheader("⚠️ Precautions")
                    for p in precautions:
                        st.write("✔", p)
                # INTERACTION
                prev_meds = get_user_meds(st.session_state.user[0])
                st.subheader("🚨 Drug Safety Check")
                warnings = check_interactions(predicted_meds, prev_meds)
                if warnings:
                    st.error("🚨 Harmful Drug Interaction Detected")
                    for w in warnings:
                        st.markdown(f"❌ **{w}**")
                else:
                    st.success("✅ No harmful interaction detected")
                # ADVICE
                st.subheader("💡 Advice")
                if conf > 0.8:
                    st.success("Follow medicines and precautions properly")
                elif conf > 0.5:
                    st.warning("Monitor your condition carefully")
                else:
                    st.error("Consult doctor immediately")
                st.info("Stay hydrated, take rest, avoid self-medication")
                st.warning("⚠️ Dosage may vary. Please consult a doctor")
                # SAVE HISTORY
                cur.execute("""
                    INSERT INTO history(user_id,symptoms,disease,medicines)
                    VALUES(?,?,?,?)
                """,(st.session_state.user[0],
                     ",".join(symptoms),
                     disease,
                     ",".join(predicted_meds)))
                conn.commit()
    # HISTORY
    elif menu == "History":

        st.title("📜 History")

        cur.execute("""
            SELECT symptoms,disease,medicines,created_at
            FROM history
            WHERE user_id=?
            ORDER BY id DESC
        """,(st.session_state.user[0],))
        rows = cur.fetchall()
        if rows:
            for r in rows:
                st.markdown(f"""
                **🦠 Disease:** {r[1]}  
                **🤒 Symptoms:** {r[0]}  
                **💊 Medicines:** {r[2]}  
                **🕒 Date:** {r[3]}  
                ---
                """)
        else:
            st.info("No history found")
    # LOGOUT
    elif menu == "Logout":
        st.session_state.user = None
        st.rerun()