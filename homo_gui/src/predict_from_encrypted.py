import pickle
import tenseal as ts
import numpy as np
from encryption.encryption_utils import get_tenseal_context

# Configuration
encrypted_input_path = "src/encrypted_patients.pkl"
context_input_path = "src/tenseal_context.ctx"
model_path = "models/trained_model.pkl"
scale_factor = 100  # Must match the one used in encryption

# Load trained model and scaler
with open(model_path, "rb") as f:
    model, scaler = pickle.load(f)

# Load TenSEAL context
with open(context_input_path, "rb") as f:
    context = ts.context_from(f.read())

# Load encrypted patient vectors
with open(encrypted_input_path, "rb") as f:
    encrypted_vectors = pickle.load(f)

# Get scaled model parameters
weights = (model.coef_[0] * scale_factor).astype(int).tolist()
bias = model.intercept_[0]

# Perform predictions
for i, serialized_vector in enumerate(encrypted_vectors):
    enc_vector = ts.bfv_vector_from(context, serialized_vector)
    enc_result = enc_vector.dot(weights)
    decrypted_score = enc_result.decrypt()[0] / (scale_factor ** 2) + bias
    predicted_class = 1 if decrypted_score >= 0.5 else 0

    print(f"\n Patient {i + 1}:")
    print(f"Decrypted Score: {decrypted_score:.4f}")
    print(f"Prediction: {'Diabetic' if predicted_class else 'Non-diabetic'}")
