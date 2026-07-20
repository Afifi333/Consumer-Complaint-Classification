# 🏦 Consumer Complaint Classification — NLP Deep Learning

An end-to-end NLP text classification pipeline that automatically predicts the correct product category from a customer's written financial complaint. Compares **SimpleRNN**, **LSTM**, **GRU**, and **fine-tuned DistilBERT** with a Gradio deployment app.

---

## 📋 Table of Contents
- [Dataset](#-dataset)
- [Complaint Categories](#-complaint-categories)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Models](#-models)
- [Results & Comparative Analysis](#-results--comparative-analysis)
- [Deployment App](#-deployment-app)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)

---

## 📂 Dataset

**Source:** [Consumer Financial Protection Bureau (CFPB) — Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

| Split | Samples |
|---|---|
| Train | ~130,000 |
| Validation / Test | ~32,483 |

- **Input:** Free-text complaint narrative written by the consumer
- **Output:** Product category (5 classes)
- **Class imbalance:** Handled via stratified splits

---

## 🏷️ Complaint Categories

| # | Class | Description |
|---|---|---|
| 0 | **Credit Card** | Disputes, unauthorized charges, billing errors |
| 1 | **Credit Reporting** | Inaccurate reports, identity theft, disputes |
| 2 | **Debt Collection** | Harassment, wrong-party contact, invalid debt |
| 3 | **Mortgages & Loans** | Loan modifications, foreclosure, servicing issues |
| 4 | **Retail Banking** | Account issues, unauthorized access, fee disputes |

---

## 📁 Project Structure

```
project nlp/
│
├── notebooks/                              # All training notebooks (run in order)
│   ├── 01_EDA_Cleaning.ipynb               #  → Data exploration & text cleaning
│   ├── 02-preprocessing-rnn.ipynb          #  → Tokenization, padding, data prep
│   ├── 03_simpleRNN.ipynb                  #  → SimpleRNN training & evaluation
│   ├── 04-lstm.ipynb                       #  → LSTM training & evaluation
│   ├── 05_GRU.ipynb                        #  → GRU training & evaluation
│   ├── 06_DistilBERT.ipynb                 #  → DistilBERT fine-tuning (HuggingFace)
│   └── 07_Model_Comparison.ipynb           #  → Side-by-side comparison & ranking
│
├── results/                                # Evaluation outputs
│   ├── simplernn_results.csv               #  → SimpleRNN metrics
│   ├── lstm_results.csv                    #  → LSTM metrics
│   ├── gru_results.csv                     #  → GRU metrics
│   ├── distilbert_results.csv              #  → DistilBERT metrics
│   ├── distilbert_classification_report.csv#  → Per-class DistilBERT report
│   ├── model_comparison.csv                #  → Final ranked comparison table
│   ├── simplernn_history.pkl               #  → SimpleRNN training history
│   ├── lstm_history.pkl                    #  → LSTM training history
│   └── gru_history.pkl                     #  → GRU training history
│
├── artifacts/                              # Reusable inference artifacts
│   ├── tokenizer.pkl                       #  → Fitted Keras tokenizer (vocab=20k)
│   ├── label_encoder.pkl                   #  → LabelEncoder for RNN models
│   └── bert_label_encoder.pkl              #  → LabelEncoder for DistilBERT
│
├── models/                                 # (gitignored — download separately)
│   ├── best_simplernn.keras                #  → Best SimpleRNN weights
│   ├── best_lstm.keras                     #  → Best LSTM weights
│   ├── best_gru.keras                      #  → Best GRU weights
│   └── distilbert_saved/                   #  → HuggingFace saved DistilBERT
│
├── data/                                   # (gitignored — download separately)
│   ├── cleaned_complaints.csv              #  → Preprocessed dataset (98 MB)
│   ├── X_train.npy / X_test.npy           #  → Padded sequences
│   └── y_train.npy / y_test.npy           #  → Labels
│
├── app.py                                  # ⭐ Gradio deployment app
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔁 Pipeline Overview

```
Raw Dataset
    ↓
01 EDA & Cleaning
  • Lowercasing
  • Remove punctuation, numbers, special characters
  • Remove stopwords
  • Class distribution analysis
    ↓
02 Preprocessing (RNN branch)
  • LabelEncoder → integer class labels
  • Stratified train/test split (80/20)
  • Keras Tokenizer (vocab_size=20,000, OOV token)
  • Sequence padding (max_len=240, post-padding)
  • Save X_train.npy, X_test.npy, tokenizer.pkl
    ↓
03-05 RNN Models (SimpleRNN / LSTM / GRU)
  • Embedding layer (dim=128)
  • Recurrent layers → Dense → Softmax
  • EarlyStopping, ReduceLROnPlateau
    ↓
06 DistilBERT (HuggingFace)
  • AutoTokenizer (distilbert-base-uncased, max_len=256)
  • AutoModelForSequenceClassification (5 labels)
  • HuggingFace Trainer API
  • Full fine-tuning on GPU (Kaggle T4)
    ↓
07 Comparison → Best Model → Deployment
```

---

## 🧠 Models

### Text Preprocessing (RNN branch)
| Parameter | Value |
|---|---|
| Vocabulary size | 20,000 |
| Max sequence length | 240 tokens |
| Padding strategy | Post-padding, post-truncation |
| OOV token | `<OOV>` |

### 1. SimpleRNN
- **Embedding** → **SimpleRNN(64)** → Dense(64, relu) → Dropout → Dense(5, softmax)
- Fast training but poor at capturing long-range dependencies

### 2. LSTM
- **Embedding** → **LSTM(128)** → Dense(64, relu) → Dropout → Dense(5, softmax)
- Gated memory cells effectively handle long complaint texts

### 3. GRU
- **Embedding** → **GRU(128)** → Dense(64, relu) → Dropout → Dense(5, softmax)
- Comparable to LSTM with fewer parameters and faster convergence

### 4. DistilBERT (Fine-tuned) ⭐ Best
- **Base model:** `distilbert-base-uncased` (HuggingFace)
- **Max length:** 256 tokens
- **Architecture:** DistilBERT encoder → classification head (5 classes)
- **Strategy:** Full fine-tuning (all layers trainable)
- **Hardware:** Kaggle T4 GPU

---

## 📊 Results & Comparative Analysis

### Overall Metrics

| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| 1 | **DistilBERT** ⭐ | **89.72%** | **87.34%** | **85.66%** | **86.47%** |
| 2 | GRU | 88.37% | 84.70% | 85.16% | 84.91% |
| 3 | LSTM | 88.21% | 84.56% | 84.88% | 84.69% |
| 4 | SimpleRNN | 76.11% | 60.27% | 63.66% | 59.18% |

### DistilBERT Per-Class Results (Best Model)

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Credit Card | 82.4% | 81.1% | 81.8% | 3,113 |
| Credit Reporting | 92.5% | 95.2% | **93.8%** | 18,235 |
| Debt Collection | 84.4% | 79.7% | 81.9% | 4,630 |
| Mortgages & Loans | 88.5% | 83.0% | 85.6% | 3,798 |
| Retail Banking | 88.9% | 89.4% | 89.1% | 2,707 |
| **Macro Avg** | **87.3%** | **85.7%** | **86.5%** | 32,483 |

### Discussion

**SimpleRNN** significantly underperforms (Macro F1: 59.2%) due to its inability to capture long-range dependencies. Consumer complaint narratives are typically long and complex — SimpleRNN's hidden state collapses for sequences > 100 tokens, leading to poor feature extraction.

**LSTM and GRU** achieve nearly identical performance (~88% accuracy, ~84.7% Macro F1). GRU marginally outperforms LSTM (88.37% vs 88.21%) despite having fewer parameters, demonstrating more efficient gradient flow on this dataset. Both benefit from gating mechanisms that selectively retain relevant context across long sequences.

**DistilBERT** achieves the best performance across all metrics (89.72% accuracy, 86.47% Macro F1), benefiting from pretrained contextual word representations trained on a massive corpus. It particularly excels on *Credit Reporting* (F1: 93.8%), which has the most training examples, and *Retail Banking* (F1: 89.1%). The smallest gap between classes reflects DistilBERT's superior generalization from pretraining.

**Key Takeaways:**
- Pretrained transformers significantly outperform task-specific RNN architectures
- GRU is a better choice than LSTM when compute/training time is a constraint
- SimpleRNN is unsuitable for production-grade long-text classification

---

## 🚀 Deployment App

A **Gradio** interactive web app (`app.py`) allows users to:

1. **Select a model** — SimpleRNN, LSTM, GRU, or DistilBERT (Best)
2. **Enter a complaint narrative** — free-text input
3. **Get instant prediction** — predicted category + confidence score
4. **View per-category probabilities** — ranked confidence bar chart
5. **Browse example complaints** — pre-loaded examples to test quickly
6. **Compare all models** — built-in comparison tab with full metrics

### Run the App

```bash
pip install -r requirements.txt
python app.py
```

> ⚠️ **Model files required:** Place `best_simplernn.keras`, `best_lstm.keras`,
> `best_gru.keras`, `tokenizer.pkl`, `label_encoder.pkl`, `bert_label_encoder.pkl`
> in the same directory. For DistilBERT, export the saved model to `distilbert_saved/`.

---

## 🔧 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Afifi333/<repo-name>.git
cd "project nlp"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model artifacts

Download the model files separately (excluded from git due to size) and place them in the project root.

### 4. Launch the Gradio app

```bash
python app.py
```

### 5. Re-train on Kaggle (optional)

Run notebooks `01` through `07` in order on Kaggle with GPU enabled. Each notebook saves its outputs to `/kaggle/working/`.

---

## 📦 Requirements

```
tensorflow>=2.15.0
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
gradio>=4.26.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
```

---

## 👤 Author

**Afifi333** — [GitHub](https://github.com/Afifi333)

---

## 📄 License

This project is for academic / educational purposes.
