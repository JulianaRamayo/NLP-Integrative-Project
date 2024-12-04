from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfLogRegClassifier():
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(random_state=42)

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.__dict__
    
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__ = state

    def train(self, x_text, y_labels):
        """
        Train the classifier on the given data.

        Args:
            x_text (list): List of text strings.
            y_labels (list): List of labels.
        """
        x_tfidf = self.vectorizer.fit_transform(x_text)
        self.model.fit(x_tfidf, y_labels)
        
        return self.model.score(x_tfidf, y_labels)
    
    def predict(self, x_text):
        """
        Evaluate sentiment on given data.

        Args:
            x_text (list): List of text strings.
        """
        x_tfidf = self.vectorizer.transform(x_text)

        return self.model.predict(x_tfidf)
    
    def evaluate(self, x_text, y_true):
        """
        Evaluate the model and return a classification report.

        Args:
            x_text (list): List of text strings.
            y_true (list): List of true labels.
        """

        predictions = self.predict(x_text)

        return classification_report(y_true, predictions)