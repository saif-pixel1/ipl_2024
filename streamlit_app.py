import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
columns = joblib.load("columns.pkl")

st.title("IPL Match Winner Predictor")

# Example inputs (simplified)
team1 = st.text_input("Team 1")
team2 = st.text_input("Team 2")
venue = st.text_input("Venue")
month = st.selectbox("Month", [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
])

if st.button("Predict"):
    # Create input dict (VERY IMPORTANT)
    input_dict = {col: 0 for col in columns}
    
    # Example encoding (adjust based on your columns)
    if f"team1_{team1}" in input_dict:
        input_dict[f"team1_{team1}"] = 1
        
    if f"team2_{team2}" in input_dict:
        input_dict[f"team2_{team2}"] = 1
        
    if f"venue_{venue}" in input_dict:
        input_dict[f"venue_{venue}"] = 1
        
    if f"month_{month}" in input_dict:
        input_dict[f"month_{month}"] = 1

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)
    result = le.inverse_transform(prediction)

    st.success(f"Predicted Winner: {result[0]}")