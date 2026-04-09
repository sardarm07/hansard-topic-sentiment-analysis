# Hansard Topic Modeling and Sentiment Analysis

NLP analysis of UK parliamentary (Hansard) debate transcripts using topic modeling and sentiment analysis to explore themes, emotional patterns, and political discourse over time.

## Overview

This project analyzes UK parliamentary debate transcripts from the Hansard dataset to:

- Identify and track discussion topics across parliamentary sessions
- Analyze sentiment patterns in political speeches
- Correlate topics with sentiment trends over time
- Compare topic distributions across political parties and speakers
- Evaluate pre-trained sentiment models against ground truth labels

## Features

### Text Preprocessing
- Stop word removal, punctuation cleaning, lowercasing
- Tokenization with stemming and lemmatization

### Topic Modeling
- **LDA** (Latent Dirichlet Allocation) with coherence-based hyperparameter tuning and pyLDAvis visualization
- **BERTopic** for transformer-based topic extraction
- Dynamic topic modeling to track topic evolution over time
- Topic distribution analysis across political parties and speakers
- Automated topic labeling using cosine similarity with predefined labels

### Sentiment Analysis
- Multi-model sentiment classification combining AFINN, Bing, NRC, SentiWord, and Hu-Liu scores
- Weighted ensemble with confidence and agreement metrics
- Word cloud and n-gram analysis for positive/negative speeches
- Feature-based analysis by `party_group` and `gender`
- Correlation heatmaps across sentiment models

### Sentiment Prediction
- Word2Vec (GoogleNews-300) embeddings with Logistic Regression
- DistilBERT fine-tuning for sentiment classification
- Evaluation against ground truth labels (MSE, RMSE)
- Classifier comparison: Logistic Regression, SVM, Random Forest, LSTM, CNN

## Project Structure

```
.
├── scripts/
│   ├── text_preprocessing.py       # Text cleaning and normalization
│   ├── data_exploration.py         # Feature distribution analysis
│   ├── sentiment_analysis.py       # Sentiment classification & visualization
│   ├── sentiment_correlation.py    # Sentiment-feature correlation
│   ├── sentiment_prediction.py     # ML-based sentiment prediction
│   ├── sentiment_model_comparison.py # Model comparison
│   ├── topic_modeling.py           # LDA & BERTopic implementation
│   ├── task6.py                    # Topic evolution over time
│   └── llm_exploration.py         # LLM-based analysis
├── graphs/                         # Generated visualizations
├── outputT6/                       # Topic evolution outputs
├── outputT7/                       # Sentiment-topic correlation outputs
├── requirements.txt
└── README.md
```

## Tech Stack

- **Language:** Python 3.8+
- **Topic Modeling:** BERTopic, Gensim (LDA), pyLDAvis
- **Sentiment:** VADER, TextBlob, AFINN, DistilBERT
- **Embeddings:** sentence-transformers, Word2Vec, transformers
- **ML:** scikit-learn, PyTorch
- **Visualization:** matplotlib, seaborn, WordCloud

## Installation

```bash
git clone https://github.com/sardarm07/Topic-Modelling-and-Sentiment-Analysis-Hansard.git
cd Topic-Modelling-and-Sentiment-Analysis-Hansard
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Place the Hansard dataset (`senti_df.csv`) in the `data/` directory
2. Run the analysis scripts individually or through the main notebook:

```bash
# Text preprocessing
python scripts/text_preprocessing.py

# Topic modeling
python scripts/topic_modeling.py

# Sentiment analysis
python scripts/sentiment_analysis.py

# Sentiment prediction
python scripts/sentiment_prediction.py
```

## Dataset

The project uses UK Hansard parliamentary debate transcripts containing:
- Speech text and metadata
- Speaker information (name, gender, party)
- Pre-computed sentiment scores (AFINN, Bing, NRC, SentiWord, Hu-Liu)
- Temporal data (date, year, time)

## License

This project was developed as part of academic coursework.
