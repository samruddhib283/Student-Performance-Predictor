import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load CSV dataset
df = pd.read_csv("student_scores.csv")

# Train model
X = df[['Hours_Studied', 'Previous_Score', 'Attendance_Percentage']]
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("📘 Student Performance Predictor")
st.write("Enter student details below to predict the **Final Exam Score** 📊")

# Inputs
hours = st.slider("📚 Hours Studied", min_value=0.0, max_value=12.0, step=0.5, value=5.0)
previous_score = st.number_input("📝 Previous Test Score", min_value=0.0, max_value=100.0, value=60.0)
attendance = st.slider("📅 Attendance Percentage", min_value=0.0, max_value=100.0, step=1.0, value=80.0)

# Predict
if st.button("🎯 Predict Final Score"):
    input_data = pd.DataFrame([[hours, previous_score, attendance]],
                              columns=['Hours_Studied', 'Previous_Score', 'Attendance_Percentage'])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Final Score: **{prediction:.2f}** out of 100")

# Show dataset (optional)
with st.expander("🔍 View Training Dataset"):
    st.dataframe(df)
