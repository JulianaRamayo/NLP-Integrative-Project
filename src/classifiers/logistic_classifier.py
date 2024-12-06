import pandas as pd  # For handling data
from sklearn.feature_extraction.text import CountVectorizer  # For text vectorization
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For evaluation
from joblib import Parallel, delayed  # For parallel processing
from tqdm import tqdm  # For progress visualization
import matplotlib.pyplot as plt  # For plotting
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

def clean_text(text, stop_words, char_filter, lemmatizer):
    """Clean text data."""
    # Replace this with the actual cleaning logic
    return text.lower()  # Example placeholder implementation


def preprocess_reviews(reviews, stop_words, char_filter, lemmatizer):
    """Preprocess a list of reviews."""
    print("Preprocessing reviews...")
    return Parallel(n_jobs=-1)(
        delayed(clean_text)(review, stop_words, char_filter, lemmatizer)
        for review in tqdm(reviews, desc="Preprocessing Reviews")
    )

def train_model(X_train, y_train):
    """Train a logistic regression model."""
    print("Training logistic regression model...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_model(classifier, X_test, y_test):
    """Evaluate the model using a confusion matrix."""
    print("Evaluating the model...")
    predictions = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix for Logistic Regression")
    plt.show()
    return predictions

def predict_new_reviews(classifier, new_reviews, stop_words, char_filter, lemmatizer):
    """Predict sentiment for new reviews."""
    print("Preprocessing new reviews...")
    preprocessed_reviews = preprocess_reviews(new_reviews, stop_words, char_filter, lemmatizer)
    print("Predicting sentiments for new reviews...")
    predictions = classifier.predict(preprocessed_reviews)
    results = []
    for original_review, processed_review, prediction in zip(new_reviews, preprocessed_reviews, predictions):
        sentiment = "Negative" if prediction == 1 else "Positive"
        results.append({
            "Original Review": original_review,
            "Processed Review": processed_review,
            "Predicted Sentiment": sentiment
        })
    return results
