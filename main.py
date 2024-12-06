from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load Models
logistic_model = joblib.load("./models/logistic_model.pkl")
dnn_model = load_model("./models/dnn_model.h5")
fine_tuned_model = T5ForConditionalGeneration.from_pretrained("./models/fine_tunned_model.pt")
enc_dec_model = load_model("./models/enc-dec_model.h5")
tokenizer = joblib.load("./models/tokenizer.pkl")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fine_tuned_model.to(device)

# Input model schema
class ReviewInput(BaseModel):
    text: str

# Polarity Endpoint
@app.post("/polarity")
async def predict_polarity(review: ReviewInput):
    try:
        text = review.text

        # Logistic Regression Prediction
        vectorized_text = tokenizer.transform([text])
        logistic_pred = logistic_model.predict(vectorized_text)[0]
        logistic_confidence = max(logistic_model.predict_proba(vectorized_text)[0])

        # DNN Prediction
        dnn_pred = dnn_model.predict(vectorized_text)
        dnn_confidence = np.max(dnn_pred)

        return {
            "logistic_model": {"polarity": int(logistic_pred), "confidence": float(logistic_confidence)},
            "dnn_model": {"polarity": int(dnn_pred.argmax()), "confidence": float(dnn_confidence)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate Endpoint
@app.post("/generate")
async def generate_review(review: ReviewInput):
    try:
        text = review.text

        # Logistic Regression Polarity
        vectorized_text = tokenizer.transform([text])
        polarity = "positive" if logistic_model.predict(vectorized_text)[0] == 1 else "negative"

        # Generate Inverse Polarity Text with Fine-Tuned Model
        input_text = f"transform to {'negative' if polarity == 'positive' else 'positive'}: {text}"
        input_ids = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = fine_tuned_model.generate(input_ids, max_length=128)
        generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "original_polarity": polarity,
            "generated_text": generated_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
