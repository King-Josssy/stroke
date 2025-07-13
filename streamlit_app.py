import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load('stroke_model.pkl')

# UI setup
st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Stroke Prediction App")
st.markdown("Enter patient details below to predict the risk of stroke.")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Predict button
if st.button("🩺 Predict Stroke Risk"):
    try:
        # Prepare input
        input_data = pd.DataFrame([{
            'id': 0,
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Show results
        st.subheader("🔍 Prediction Result")
        st.write(f"**Prediction:** {'⚠️ Stroke Risk' if prediction == 1 else '✅ No Stroke Risk'}")
        st.write(f"**Probability of Stroke:** {probability:.2%}")

        # Show chart
        st.subheader("📊 Glucose & BMI Visual Check")
        fig, ax = plt.subplots()
        ax.bar(["Glucose Level", "BMI"], [avg_glucose_level, bmi], color=["skyblue", "salmon"])
        ax.set_ylabel("Value")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load('stroke_model.pkl')

# UI setup
st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Stroke Prediction App")
st.markdown("Enter patient details below to predict the risk of stroke.")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Predict button
if st.button("🩺 Predict Stroke Risk"):
    try:
        # Prepare input
        input_data = pd.DataFrame([{
            'id': 0,
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Show results
        st.subheader("🔍 Prediction Result")
        st.write(f"**Prediction:** {'⚠️ Stroke Risk' if prediction == 1 else '✅ No Stroke Risk'}")
        st.write(f"**Probability of Stroke:** {probability:.2%}")

        # Show chart
        st.subheader("📊 Glucose & BMI Visual Check")
        fig, ax = plt.subplots()
        ax.bar(["Glucose Level", "BMI"], [avg_glucose_level, bmi], color=["skyblue", "salmon"])
        ax.set_ylabel("Value")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
