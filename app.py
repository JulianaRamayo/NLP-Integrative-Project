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

import torch




# Load Models
try:
    logistic_model = load_model("dnn_model.h5")  # Assuming this is a scikit-learn logistic regression model
except Exception as e:
    raise RuntimeError(f"Could not load logistic_model.pkl: {e}")

try:
    generation_model = torch.load("fine_tune-model.pt")  # Assuming a keras model
    generation_model.eval()
except Exception as e:
    raise RuntimeError(f"Could not load enc-dec_model.h5: {e}")

# Dummy tokenizer or preprocessing if needed for generation model
# You should replace this with the actual tokenizer/preprocessing steps
# used during training of the generative model.
tokenizer = tf.keras.preprocessing.text.Tokenizer()
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

def generate_opposite_polarity_text(original_text: str, original_polarity: int, model=generation_model) -> str:
    """
    Generate text with the opposite polarity using the generation model.
    This version uses a loaded Hugging Face model (e.g., T5) and tokenizer
    to generate the transformed text.

    Args:
        original_text (str): The original review text.
        original_polarity (int): 0 for negative, 1 for positive.

    Returns:
        str: The transformed review with opposite polarity.
    """
    # Determine opposite polarity
    opposite_polarity = 1 - original_polarity

    # Construct prompt based on opposite polarity
    if opposite_polarity == 0:
        # If opposite_polarity is 0, we want negative text
        prompt = f"Convert this review to a negative tone: {original_text}"
    else:
        # If opposite_polarity is 1, we want positive text
        prompt = f"Convert this review to a positive tone: {original_text}"

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)  # Move to the same device as the model if necessary

    # Generate output using the model
    # Adjust parameters like max_length, num_beams, etc., as needed
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
