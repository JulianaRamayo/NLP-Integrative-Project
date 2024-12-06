import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(train_df, test_df):
    """Clean and preprocess training and testing data."""
    train_df = train_df.dropna(subset=['review'])  # Remove rows with missing reviews
    test_df = test_df.dropna(subset=['review'])

    # Map labels: 2 → Positive (1), 1 → Negative (0)
    train_df['label'] = train_df['label'].map({2: 1, 1: 0}).astype(int)
    test_df['label'] = test_df['label'].map({2: 1, 1: 0}).fillna(0).astype(int)

    # Extract features and labels
    X_train, y_train = train_df['review'], train_df['label']
    X_test, y_test = test_df['review'], test_df['label']
    return X_train, y_train, X_test, y_test

def preprocess_text(X_train, X_test, vocab_size, max_length):
    """Tokenize and pad sequences."""
    X_train = X_train.fillna("").astype(str)  # Ensure all reviews are strings
    X_test = X_test.fillna("").astype(str)

    # Tokenize text data
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)

    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad sequences to ensure uniform length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    return X_train_pad, X_test_pad, tokenizer

class KerasModelGenerator:
    def __init__(self, vocab_size, embedding_dim, max_length):
        """Initialize the KerasModelGenerator."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def build(self):
        """Build a Keras Sequential model."""
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

def train_and_evaluate_with_kfolds(X, y, vocab_size, embedding_dim, max_length, n_splits=5, epochs=5, batch_size=128):
    """Train and evaluate using K-Fold cross-validation."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\nStarting Fold {fold}...")

        # Split data for the current fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build the model
        keras_generator = KerasModelGenerator(vocab_size, embedding_dim, max_length)
        model = keras_generator.build()

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

        # Evaluate the model
        y_val_pred_prob = model.predict(X_val)
        y_val_pred = (y_val_pred_prob > 0.5).astype(int)

        # Compute metrics
        fold_accuracy = accuracy_score(y_val, y_val_pred)
        fold_metrics.append(fold_accuracy)

        print(f"Fold {fold} Accuracy: {fold_accuracy:.4f}")
        print(f"Classification Report for Fold {fold}:\n", classification_report(y_val, y_val_pred, target_names=['Negative', 'Positive']))

        # Clear TensorFlow session to release memory
        from tensorflow.keras import backend as K
        K.clear_session()

    avg_metrics = {"accuracy": np.mean(fold_metrics)}
    print("\nAverage Metrics Across Folds:")
    print(avg_metrics)
    return avg_metrics

def predict_new_reviews(model, tokenizer, new_reviews, max_length):
    """Predict sentiments for new reviews."""
    # Tokenize and pad the new reviews
    new_reviews_seq = tokenizer.texts_to_sequences(new_reviews)
    new_reviews_pad = pad_sequences(new_reviews_seq, maxlen=max_length, padding='post')

    # Make predictions
    predictions_prob = model.predict(new_reviews_pad)
    predictions = (predictions_prob > 0.5).astype(int)

    # Convert to human-readable labels
    sentiments = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
    return sentiments