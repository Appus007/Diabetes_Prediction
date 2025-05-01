import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tenseal as ts
from encryption.encryption_utils import get_tenseal_context

st.set_page_config(page_title="Diabetes Homomorphic Prediction", layout="wide")
st.title("🩺 Diabetes Prediction with Homomorphic Encryption")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["🔐 Encrypt Data", "🧠 Decrypt & Predict"])

# Session state
if 'encrypted_data' not in st.session_state:
    st.session_state['encrypted_data'] = None
if 'context_data' not in st.session_state:
    st.session_state['context_data'] = None


# 🔐 Encrypt Section
if page == "🔐 Encrypt Data":
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
            if st.button("🔐 Encrypt Patient Data"):
                context = get_tenseal_context()
                context_bytes = context.serialize(save_secret_key=True)

                encrypted_vectors = []
                for _, row in df[required_features].iterrows():
                    scaled = (row * 100).astype(int).tolist()
                    vec = ts.bfv_vector(context, scaled)
                    encrypted_vectors.append(vec.serialize())

                encrypted_data = pickle.dumps(encrypted_vectors)

                st.session_state['encrypted_data'] = encrypted_data
                st.session_state['context_data'] = context_bytes

                st.success("✅ Encryption complete. You can now download the files below.")

            if st.session_state['encrypted_data'] and st.session_state['context_data']:
                st.markdown("### 🔽 Download Encrypted Files")
                st.download_button(
                    "📄 Download Encrypted Patient Data (.pkl)",
                    data=st.session_state['encrypted_data'],
                    file_name="encrypted_patients.pkl",
                    mime="application/octet-stream"
                )
                st.download_button(
                    "🔑 Download Encryption Context (.ctx)",
                    data=st.session_state['context_data'],
                    file_name="tenseal_context.ctx",
                    mime="application/octet-stream"
                )


# 🧠 Decrypt & Predict Section
elif page == "🧠 Decrypt & Predict":
    st.markdown("### 📤 Upload Encrypted Files")
    uploaded_enc_file = st.file_uploader("Encrypted Patient Data (.pkl)", type=["pkl"])
    uploaded_ctx_file = st.file_uploader("Encryption Context (.ctx)", type=["ctx"])

    if uploaded_enc_file and uploaded_ctx_file:
        try:
            encrypted_vectors = pickle.load(uploaded_enc_file)
            context_data = uploaded_ctx_file.read()
            context = ts.context_from(context_data)

            with open("models/trained_model.pkl", "rb") as f:
                model, scaler = pickle.load(f)

            weights = (model.coef_[0] * 100).astype(int).tolist()
            bias = model.intercept_[0]

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

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.info("📁 Please upload both the encrypted data (.pkl) and the encryption context (.ctx).")
