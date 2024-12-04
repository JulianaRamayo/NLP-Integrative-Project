
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.char_filter = re.compile('[^a-zA-Z\s]')

    def clean_text(self, text):
        """
        Clean and preprocess a single text string.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()

        text = self.char_filter.sub(' ', text)

        tokens = word_tokenize(text)

        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]

        return ' '.join(tokens)

    def preprocess_dataset(self, df):
        """
        Preprocess the entire dataset.
        Returns a new DataFrame with cleaned text.
        """
        processed_df = df.copy()

        processed_df[2] = processed_df[2].apply(self.clean_text)

        processed_df[1] = processed_df[1].apply(self.clean_text)

        return processed_df