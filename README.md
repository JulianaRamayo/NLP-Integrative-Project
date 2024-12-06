# Amazon Review Sentiment Analysis and Transformation System

## Project Overview

This project implements a comprehensive system for analyzing and transforming the sentiment of Amazon product reviews using advanced Natural Language Processing (NLP) techniques. The system integrates both classification and generative models to tackle sentiment analysis and transformation tasks effectively.

### Objectives:
1. **Classification Models**: Develop models to categorize reviews as positive or negative using:
   - Logistic Regression with TF-IDF vectorization.
   - Deep Neural Networks (DNNs).
2. **Generative Models**: Create models to transform reviews from positive to negative sentiment and vice versa using:
   - Sequence-to-sequence (Encoder-Decoder) architecture.
   - Fine-tuned pre-trained transformer models (T5).

---

## Dataset and Preprocessing

The **Amazon Reviews Dataset** from Kaggle was used, consisting of:
- **Training Set**: 3,599,774 reviews.
- **Testing Set**: 399,975 reviews.

### Preprocessing Steps:
- **Tokenization**: Splitting reviews into tokens for analysis.
- **Text Cleaning**: Removal of special characters, numbers, and extra spaces; conversion to lowercase.
- **Stop Words Removal**: Filtering common words like "and" or "is."
- **Sentiment Label Encoding**: Mapping reviews with 4-5 stars to positive and 1-3 stars to negative.
- **Handling Missing Data**: Excluding incomplete reviews.
- **Text Length Standardization**: Padding/truncating text to fixed lengths.

---

## Model Architectures

### Classification Models

#### Logistic Regression with TF-IDF
- Converts reviews into numerical features using TF-IDF vectorization.
- Trained with 5-fold cross-validation.
- **Performance**:
  - Average Accuracy: **87.49%**
  - Example:  
    **Original Review**: _I couldn’t stop laughing at how bad it was! The best comedy ever made._  
    **Processed Review**: _stop laughing bad best comedy ever made._  
    **Predicted Sentiment**: Positive.

#### Deep Neural Network (DNN)
- Embedding layers convert tokens into dense vectors.
- Fully connected layers with ReLU activation and dropout regularization.
- **Performance**:
  - Accuracy for Fold 1: **87.58%**
  - Example: Comparable precision and recall scores for positive and negative classes.

### Generative Models

#### Encoder-Decoder Model
- Sequence-to-sequence framework with LSTM layers for sentiment transformation.
- **Performance**: Generated outputs were grammatically correct but often lacked semantic coherence.

#### Fine-Tuned Pre-trained Model (T5)
- Hugging Face’s T5 model fine-tuned for sentiment transformation.
- **Performance**:
  - Perplexity: **1.19**
  - Example:  
    **Original**: _Very good movie._  
    **Transformed**: _movie._

---

## Experiment Results

### Classification Results
| Model                   | Average Accuracy | Remarks                              |
|-------------------------|------------------|--------------------------------------|
| Logistic Regression     | 87.49%          | Reliable and computationally efficient. |
| Deep Neural Network     | 87.58%          | Captures complex patterns; scalable. |

### Generative Model Results
| Model                   | Key Metric       | Remarks                              |
|-------------------------|------------------|--------------------------------------|
| Encoder-Decoder         | -                | Outputs lacked coherence.            |
| Fine-Tuned Pre-trained  | Perplexity: 1.19 | Accurate and efficient sentiment transformations. |

---

## API Endpoints

1. **Polarity Detection**:  
   Input: Review text.  
   Output: Polarity (0 = Negative, 1 = Positive) with confidence score.

2. **Sentiment Transformation**:  
   Input: Review text.  
   Output: JSON with original polarity and transformed text.

---

## Repository and Report

### Deliverables:
1. **Source Code**: Available on GitHub [Link].
2. **Report**: Detailed PDF document explaining methodology, results, and conclusions.

### Deadline:
Submission by **05/12/2024 13:59:59** (Moodle Time).

---

## Conclusion

This project demonstrates the potential of combining classification and generative NLP techniques for sentiment analysis and transformation. Future work could include scaling models to larger datasets, integrating advanced transformers, and refining generative architectures for better fluency and coherence.