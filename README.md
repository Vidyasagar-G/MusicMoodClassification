# Music Mood Classification using Lyrics

This repository contains the code and experimental results for a DA623 (Winter 2025) course project at IIT Guwahati, focused on **classifying the emotional mood of songs using their lyrics**. The broader research motivation is to explore **multimodal music mood classification** — combining both **textual** (lyrics) and **acoustic** (audio) cues to model musical emotion. While audio integration remains a work in progress, this report presents a strong lyrics-only baseline.

## Project Overview

The goal of this project is to understand whether song lyrics alone can be used to accurately predict a song’s mood. We approached this as a **multi-class classification problem**, where each song is labeled as one of the following:

- Angry
- Happy
- Relaxed
- Sad

The workflow includes:

- **Lyrics preprocessing** using NLTK (lemmatization, stopword removal)
- **Text feature extraction** using:
  - TF-IDF
  - GloVe (100-dimensional average word embeddings)
  - BERT-based sentence embeddings (MiniLM)
- **Model training** using:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Random Forest
  - Naive Bayes (TF-IDF only)
- **Evaluation** using:
  - Accuracy
  - Per-class F1-scores
  - Confusion matrices

## Dataset

We used a subset of the [NJU Music Mood Dataset](https://cs.nju.edu.cn/sufeng/data/musicmood.htm), which provides timestamped song lyrics labeled across four mood categories. Only the lyrics were available for use, due to copyright restrictions.

- Total samples: 777
- Train/Test split: 400 / 377
- Each sample is a `.txt` file containing timestamped lyrics.

> Note: Audio features were intended to be used in this project but were not included due to data access limitations. Future work aims to incorporate acoustic signals.

## Key Findings

- All models plateaued around ~40% accuracy, suggesting that lyrics alone may not provide enough signal for accurate mood classification.
- BERT embeddings did not significantly outperform simpler TF-IDF representations.
- Mood classes such as "Relaxed" and "Sad" were frequently confused, even by advanced models.
- These limitations reinforce the need for multimodal learning in music-based tasks.

## Files

- `Music Mood Classification.ipynb`: Full project notebook including preprocessing, feature extraction, modeling, evaluation, and reflections.
- `lyrics_dataset.csv`: Preprocessed dataset (one row per song) with mood labels.

## Future Work

- Incorporate audio features (e.g., MFCCs, chroma, tempo) using Librosa
- Train multimodal models that combine lyrics and audio
- Explore transformer-based multimodal fusion techniques
- Address label ambiguity by applying hierarchical or regression-based mood models
