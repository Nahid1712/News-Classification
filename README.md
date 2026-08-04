# 📰 BERT News Classifier

This project fine-tunes a **BERT-base-uncased** model to classify news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. It includes the full pipeline: data preprocessing, model training, evaluation (with metrics and confusion matrix), and an inference script for real-time predictions.

## 📊 Performance Summary

The fine-tuned model achieves excellent results on the test set:

- **Accuracy**: 94.78%
- **F1 Score (Weighted)**: 94.78%
- **F1 Score (Macro)**: 94.78%

### Per-Class Performance

| Class     | Precision | Recall  | F1-Score |
|-----------|-----------|---------|----------|
| World     | 96.07%    | 95.26%  | 95.67%   |
| Sports    | 99.00%    | 98.84%  | 98.92%   |
| Business  | 92.68%    | 91.32%  | 91.99%   |
| Sci/Tech  | 91.42%    | 93.68%  | 92.54%   |

## 🗂️ Dataset

The dataset is structured similarly to the classic **AG News** corpus and contains the following columns:

- **Class Index**: Integer labels (1 to 4).
- **Title**: The headline of the news article.
- **Description**: A short description/summary of the article.

The notebook combines `Title` and `Description` into a single `text` field using a `[SEP]` token before feeding it into BERT.

## ⚙️ Model Architecture

- **Base Model**: `bert-base-uncased` (HuggingFace Transformers).
- **Head**: A custom `nn.Linear` classifier head on top of the pooled output.
- **Training Setup**:
  - Optimizer: AdamW (learning rate = 2e-5, weight decay = 0.01).
  - Loss Function: Cross-Entropy Loss.
  - Batch Size: 16.
  - Epochs: 5.
  - Max Sequence Length: 128 tokens.

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed. The project relies heavily on the following libraries:

- `torch` (PyTorch)
- `transformers` (HuggingFace)
- `pandas`
- `scikit-learn`
- `matplotlib` & `seaborn`
- `tqdm`

### 2. Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/bert-news-classifier.git
cd bert-news-classifier
pip install -r requirements.txt 
