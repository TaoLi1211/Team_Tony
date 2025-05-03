import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('./best_model.pkl')

# Clearly load column names from your training dataset (preprocessed columns)
# You should save these columns during training; example provided below:
training_columns = joblib.load('./training_columns.pkl')

st.title('🔧 Predictive Maintenance Dashboard')

# UI for user input
type_input = st.selectbox('Equipment Type', ['L', 'M', 'H'])
air_temp = st.number_input('Air Temperature [K]', 200.0, 400.0, 298.0)
process_temp = st.number_input('Process Temperature [K]', 200.0, 400.0, 308.0)
rotational_speed = st.number_input('Rotational Speed [rpm]', 500, 3000, 1500)
torque = st.number_input('Torque [Nm]', 0, 500, 40)
tool_wear = st.number_input('Tool Wear [min]', 0, 500, 100)

# DataFrame from inputs
input_df = pd.DataFrame({
    'Type': [type_input],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear]
})

# One-hot encode categorical variables
input_df = pd.get_dummies(input_df, drop_first=True)

# Match model features exactly
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Drop any extra columns not in training data
input_df = input_df[training_columns]

# Make prediction clearly matching trained model columns
if st.button('Predict Maintenance'):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    if prediction == 0:
        st.success('✅ Equipment is healthy! No immediate maintenance required.')
        st.info(f"Confidence: {prediction_proba[0]:.2%}")
    else:
        st.error('⚠️ Maintenance required! Immediate attention recommended.')
        st.info(f"Confidence: {prediction_proba[1]:.2%}")
