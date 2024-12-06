from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import uvicorn
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Define request models
class PolarityRequest(BaseModel):
    review: str

class GenerateRequest(BaseModel):
    review: str

# Initialize app
app = FastAPI(
    title="Review Polarity API",
    description="Endpoints to determine polarity and generate opposite polarity text."
)

from model_definitions import ReviewClassifier


# Load Models
try:
    logistic_model = load_model("dnn_model.h5")  # Assuming this is a scikit-learn logistic regression model
except Exception as e:
    raise RuntimeError(f"Could not load logistic_model.pkl: {e}")

try:
    generation_model = load_model("fine_tunned-model.h5")  # Assuming a keras model
except Exception as e:
    raise RuntimeError(f"Could not load fine_tunned-model.h5: {e}")

# Dummy tokenizer or preprocessing if needed for generation model
# You should replace this with the actual tokenizer/preprocessing steps
# used during training of the generative model.
tokenizer = None 
# Example: tokenizer = tf.keras.preprocessing.text.Tokenizer(...)
# tokenizer.fit_on_texts(...)  # if you have a fixed vocabulary

def predict_polarity(text: str) -> (int, float):
    """
    Predict polarity using the logistic regression model.
    Returns:
      polarity: int (0 or 1)
      confidence: float (probability of the predicted class)
    """
    # Logistic regression usually expects some feature extraction before prediction.
    # This example assumes your logistic_model already handles raw text internally 
    # (maybe via a pipeline with TF-IDF). If not, you must apply the same preprocessing
    # steps here that you did during training.
    proba = logistic_model.predict_proba([text])[0]  # e.g. [neg_prob, pos_prob]
    polarity = int(np.argmax(proba))  # 0 for negative, 1 for positive
    confidence = float(np.max(proba))
    return polarity, confidence

def generate_opposite_polarity_text(original_text: str, original_polarity: int) -> str:
    """
    Generate text with the opposite polarity using the generation model.
    This is a placeholder function. You must implement it according to how
    your generation model is intended to be used.
    
    original_polarity: 0 or 1
    opposite_polarity: If original_polarity=0 (negative), we want positive text.
                       If original_polarity=1 (positive), we want negative text.
    """
    opposite_polarity = 1 - original_polarity

    # Example prompt construction:
    # If the original text is positive, we want to generate a negative version.
    # If it is negative, we want to generate a positive version.
    if opposite_polarity == 0:
        prompt = f"Convert this review to a negative tone: {original_text}"
    else:
        prompt = f"Convert this review to a positive tone: {original_text}"

    # The following is a placeholder. If your model is a seq2seq, you may need to:
    # 1. Tokenize the prompt
    # 2. Use model.predict() or a custom generation method (e.g. greedy search, beam search)
    # 3. Convert generated token indices back to text.

    # Placeholder logic (replace with actual generation code):
    # Example: assume a seq2seq model that takes a tokenized prompt and returns token IDs
    # tokenized = tokenizer.texts_to_sequences([prompt])
    # padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=MAX_LEN)
    # generated_ids = generation_model.predict(padded)
    # generated_text = tokenizer.sequences_to_texts(generated_ids)
    # return generated_text[0]

    # Since this is a placeholder, let's just return a mock string:
    generated_text = f"[Generated { 'negative' if opposite_polarity == 0 else 'positive'} text for: '{original_text}']"

    return generated_text


@app.post("/polarity", summary="Predict polarity of a given review text.")
def polarity_endpoint(request: PolarityRequest) -> Dict:
    if not request.review:
        raise HTTPException(status_code=400, detail="Review text is required.")
    polarity, confidence = predict_polarity(request.review)
    return {"polarity": polarity, "confidence": confidence}


@app.post("/generate", summary="Generate text with opposite polarity.")
def generate_endpoint(request: GenerateRequest) -> Dict:
    if not request.review:
        raise HTTPException(status_code=400, detail="Review text is required.")
    original_polarity, _ = predict_polarity(request.review)
    opposite_text = generate_opposite_polarity_text(request.review, original_polarity)
    # Determine polarity strings (for readability)
    polarity_str = "positive" if original_polarity == 1 else "negative"
    return {"original_polarity": polarity_str, "generated_text": opposite_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
