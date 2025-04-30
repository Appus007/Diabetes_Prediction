import sys
import os

# Add project root to sys.path so 'encryption' and 'src' modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tenseal as ts
from encryption.encryption_utils import get_tenseal_context




# Title
st.title("🩺 Diabetes Prediction with Homomorphic Encryption")

# Upload CSV
uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
    st.dataframe(df)

    required_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    if not all(col in df.columns for col in required_features):
        st.error("CSV is missing required features.")
    else:
        # Encrypt button
        if st.button("🔐 Encrypt Patient Data"):
            context = get_tenseal_context()

            # Save context with secret key (for local use only)
            with open("tenseal_context.ctx", "wb") as f:
                f.write(context.serialize(save_secret_key=True))

            encrypted_vectors = []
            for _, row in df[required_features].iterrows():
                scaled = (row * 100).astype(int).tolist()
                vec = ts.bfv_vector(context, scaled)
                encrypted_vectors.append(vec.serialize())

            # Save encrypted data
            with open("encrypted_patients.pkl", "wb") as f:
                pickle.dump(encrypted_vectors, f)

            st.success("✅ Encrypted data and context saved!")

# Prediction
if st.button("🧠 Run Prediction on Encrypted Data"):
    try:
        # Load model
        with open("models/trained_model.pkl", "rb") as f:
            model, scaler = pickle.load(f)

        weights = (model.coef_[0] * 100).astype(int).tolist()
        bias = model.intercept_[0]

        # Load context and encrypted vectors
        with open("tenseal_context.ctx", "rb") as f:
            context = ts.context_from(f.read())
        with open("encrypted_patients.pkl", "rb") as f:
            encrypted_vectors = pickle.load(f)

        # Predict
        results = []
        for i, enc_vec_ser in enumerate(encrypted_vectors):
            enc_vec = ts.bfv_vector_from(context, enc_vec_ser)
            enc_result = enc_vec.dot(weights)
            score = enc_result.decrypt()[0] / (100 ** 2) + bias
            label = "Diabetic" if score >= 0.5 else "Non-Diabetic"
            results.append((i + 1, round(score, 4), label))

        st.subheader("🧾 Prediction Results")
        result_df = pd.DataFrame(results, columns=["Patient ID", "Score", "Prediction"])
        st.dataframe(result_df)

        # Optional download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
