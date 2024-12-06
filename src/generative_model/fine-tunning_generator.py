import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from tqdm import tqdm

def preprocess_dataframe(df):
    """
    Preprocess the DataFrame by filling NaN values, converting columns to strings, 
    and mapping polarity labels to string categories.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df['title'] = df['title'].fillna('').astype(str)
    df['text'] = df['text'].fillna('').astype(str)
    df['polarity'] = df['polarity'].apply(lambda x: 'positive' if x == 2 else 'negative')
    return df

class AmazonReviewsDataset(Dataset):
    """
    Custom Dataset class for tokenizing and preparing Amazon reviews.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        tokenizer (T5Tokenizer): Tokenizer for text processing.
        max_len (int): Maximum token length for inputs and outputs.
    """
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        polarity = 'negative' if row['polarity'] == 'positive' else 'positive'
        text = f"transform to {polarity}: {row['title']} {row['text']}"

        inputs = self.tokenizer.encode_plus(
            text=text.strip(),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        targets = self.tokenizer.encode_plus(
            text=row['text'].strip(),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def train_epoch(model, data_loader, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (T5ForConditionalGeneration): The model to train.
        data_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        device (torch.device): Device for computation (CPU/GPU).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def calculate_perplexity(model, data_loader, device):
    """
    Calculate the perplexity of the model on the dataset.

    Args:
        model (T5ForConditionalGeneration): The trained model.
        data_loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): Device for computation (CPU/GPU).

    Returns:
        float: Perplexity of the model.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(data_loader)
    return torch.exp(torch.tensor(avg_loss)).item()

def generate_review(model, tokenizer, text, device, target_polarity):
    """
    Generate a transformed review based on the target polarity.

    Args:
        model (T5ForConditionalGeneration): The trained model.
        tokenizer (T5Tokenizer): Tokenizer for text encoding/decoding.
        text (str): Original review text.
        device (torch.device): Device for computation (CPU/GPU).
        target_polarity (str): Target polarity ('positive' or 'negative').

    Returns:
        str: Transformed review.
    """
    model.eval()
    input_text = f"transform to {target_polarity}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def train_model(model, data_loader, optimizer, device, epochs=3):
    """
    Train the model for multiple epochs and calculate perplexity.

    Args:
        model (T5ForConditionalGeneration): The model to train.
        data_loader (DataLoader): DataLoader for the dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device for computation (CPU/GPU).
        epochs (int): Number of epochs.

    Returns:
        float: Perplexity after training.
    """
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss = train_epoch(model, data_loader, optimizer, device)
        print(f'Train loss: {train_loss:.4f}')

    perplexity = calculate_perplexity(model, data_loader, device)
    print(f'Perplexity: {perplexity}')
    return perplexity