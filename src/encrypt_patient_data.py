import pandas as pd
import pickle
import tenseal as ts
import os
from encryption.encryption_utils import get_tenseal_context

# Configuration
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_csv = "src/patients.csv"
encrypted_output_path = "src/encrypted_patients.pkl"
context_output_path = "src/tenseal_context.ctx"
scale_factor = 100  # Used to convert floats to integers

# Load and validate CSV
df = pd.read_csv(input_csv)
missing = set(features) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Create TenSEAL context
context = get_tenseal_context()

# Save the context to file (with secret key for local use)
with open(context_output_path, "wb") as f:
    f.write(context.serialize(save_secret_key=True))

# Encrypt each row and store serialized vectors
encrypted_vectors = []
for _, row in df[features].iterrows():
    scaled_input = (row * scale_factor).astype(int).tolist()
    encrypted_vector = ts.bfv_vector(context, scaled_input)
    encrypted_vectors.append(encrypted_vector.serialize())

# Save encrypted vectors to file
with open(encrypted_output_path, "wb") as f:
    pickle.dump(encrypted_vectors, f)

print(f"\n Encrypted patient data saved to: {encrypted_output_path}")
print(f"TenSEAL context saved to: {context_output_path}")
