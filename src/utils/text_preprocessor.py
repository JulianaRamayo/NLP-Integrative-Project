
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pandas as pd
class TextPreprocessor:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words("english"))
        self.char_filter = re.compile('[^a-zA-Z\s]')

    def clean_text(self, text:str) -> str:
        """
        Clean and preprocess a text string.

        Args:
            text (str): Text string to clean.        
        """

        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        
        text = text.lower()
        text = self.char_filter.sub('', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopwords]

        return ' '.join(tokens)
    
    def preprocess_dataset(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire dataset.
        Returns a new DataFrame with cleaned text.

        Args:
            df (pd.DataFrame): DataFrame to preprocess.
        """

        processed_df = df.copy()
        processed_df[2] = processed_df[2].apply(self.clean_text)
        processed_df[1] = processed_df[1].apply(self.clean_text)

        return processed_df