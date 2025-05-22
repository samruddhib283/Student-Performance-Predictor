import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("📘 Student Performance Predictor")
st.markdown("Predict a student’s **Final Exam Score** based on their study hours, previous scores, and attendance 📊")

# Load dataset
df = pd.read_csv("student_scores.csv")

# Train model
X = df[['Hours_Studied', 'Previous_Score', 'Attendance_Percentage']]
y = df['Final_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# --- Sidebar ---
st.sidebar.header("🔧 Settings")
show_data = st.sidebar.checkbox("Show training dataset")
show_info = st.sidebar.checkbox("Show model details", value=True)

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Enter Student Details")
    hours = st.slider("📚 Hours Studied", min_value=0.0, max_value=12.0, step=0.5, value=5.0)
    previous_score = st.number_input("📝 Previous Test Score", min_value=0.0, max_value=100.0, value=60.0)
    attendance = st.slider("📅 Attendance Percentage", min_value=0.0, max_value=100.0, step=1.0, value=80.0)

    if st.button("🎯 Predict Final Score"):
        input_data = pd.DataFrame([[hours, previous_score, attendance]],
                                  columns=['Hours_Studied', 'Previous_Score', 'Attendance_Percentage'])
        prediction = model.predict(input_data)[0]
        prediction = np.clip(prediction, 0, 100)  # Ensure it stays in [0, 100]
        st.success(f"✅ Predicted Final Score: **{prediction:.2f}** out of 100")


with col2:
    st.subheader("📈 Prediction Overview")
    st.markdown("Model trained using **Linear Regression** on past performance data.")
    if show_info:
        st.code("""
Features used:
- Hours Studied
- Previous Score
- Attendance Percentage

Model: LinearRegression()
        """, language="python")

# --- Show data ---
if show_data:
    st.subheader("🔍 Training Dataset")
    st.dataframe(df)
