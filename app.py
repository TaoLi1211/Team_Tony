import streamlit as st
import pandas as pd
import joblib
from model import predict_from_input

# Load trained model and columns
model = joblib.load('model.joblib')
model_columns = joblib.load('model_columns.joblib')

st.title("Predictive Maintenance Dashboard")

st.sidebar.header("Enter Equipment Details")

# User inputs
type_input = st.sidebar.selectbox('Type', ['L', 'M', 'H'])
air_temp = st.sidebar.number_input('Air temperature [K]', value=298.0)
process_temp = st.sidebar.number_input('Process temperature [K]', value=310.0)
rot_speed = st.sidebar.number_input('Rotational speed [rpm]', value=1500.0)
torque = st.sidebar.number_input('Torque [Nm]', value=40.0)
tool_wear = st.sidebar.number_input('Tool wear [min]', value=100.0)

# Construct user input dataframe
input_data = pd.DataFrame({
    'Type': [type_input],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rot_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
})

# One-hot encode categorical inputs clearly
input_data_encoded = pd.get_dummies(input_data, columns=['Type'], drop_first=True)

# Ensure input_data_encoded contains exactly the model columns
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0  # Fill missing columns with zeros clearly

# Reorder columns explicitly to match model columns
input_data_encoded = input_data_encoded[model_columns]

# Prediction button
if st.button('Predict'):
    prediction, prediction_prob = predict_from_input(model, input_data_encoded)

    # Meaningful interpretation of prediction
    if prediction[0] == 0:
        result_text = "✅ Equipment is Healthy"
    else:
        result_text = "⚠️ Equipment Needs Maintenance"

    st.subheader("Prediction Result:")
    st.write(f"**{result_text}**")

    st.subheader("Prediction Probabilities:")
    prob_df = pd.DataFrame(prediction_prob, columns=["Healthy (0)", "Maintenance Required (1)"])
    st.write(prob_df)

    st.success('Prediction completed successfully!')
