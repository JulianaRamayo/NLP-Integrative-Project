import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import random  # For sampling
import warnings
warnings.filterwarnings("ignore")  # Optional: Suppress warnings
import nltk
nltk.download('stopwords')
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.char_filter = re.compile('[^a-zA-Z\s]')
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean and preprocess a text string."""
        if not isinstance(text, str):
            text = ""

        text = text.lower()  # Convert to lowercase
        text = self.char_filter.sub('', text)  # Remove special characters
        tokens = word_tokenize(text)  # Tokenize
        tokens = [token for token in tokens if token not in self.stop_words]  # Remove stopwords
        return ' '.join(tokens)

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the entire dataset."""
        processed_df = df.copy()
        processed_df[2] = processed_df[2].apply(self.clean_text)
        processed_df[1] = processed_df[1].apply(self.clean_text)
        return processed_df

def prepare_dataset(sample_data, max_length=200):
    """Prepare input and target text datasets."""
    input_texts = sample_data[sample_data['polarity'] == 2]['text']
    target_texts = sample_data[sample_data['polarity'] == 1]['text']

    input_texts = input_texts.str[:max_length].reset_index(drop=True)
    target_texts = target_texts.str[:max_length].apply(lambda x: '\t' + x + '\n').reset_index(drop=True)

    return input_texts, target_texts

def build_character_sets(input_texts, target_texts):
    """Build character sets from input and target texts."""
    input_characters = sorted(set(''.join(input_texts)))
    target_characters = sorted(set(''.join(target_texts)))
    return input_characters, target_characters

def encode_data(input_texts, target_texts, input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length):
    """One-hot encode input and target texts."""
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index

def build_and_train_model(encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens, latent_dim=256, batch_size=64, epochs=15):
    """Build and train the sequence-to-sequence model."""
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model

def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_decoder_seq_length, temperature=1.0):
    """Decode a sequence using the trained model."""
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, len(target_token_index)))
    target_seq[0, 0, target_token_index['\t']] = 1.0

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, len(target_token_index)))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence