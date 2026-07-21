import gradio as gr
import numpy as np
import pickle
import re
import os

# ── Try to load TensorFlow models (RNN-based) ──────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Try to load HuggingFace (DistilBERT) ──────────────────────────
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_LEN    = 240
VOCAB_SIZE = 20000

CLASS_LABELS = [
    "Credit Card",
    "Credit Reporting",
    "Debt Collection",
    "Mortgages & Loans",
    "Retail Banking",
]

# Internal class names matching label_encoder order
CLASS_NAMES_INTERNAL = [
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "mortgages_and_loans",
    "retail_banking",
]

# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

def load_pkl(fname, subdir="artifacts"):
    path = os.path.join(BASE_DIR, subdir, fname)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

tokenizer      = load_pkl("tokenizer.pkl")
label_encoder  = load_pkl("label_encoder.pkl")
bert_le        = load_pkl("bert_label_encoder.pkl")

# Load Keras models
def load_keras(fname):
    if not TF_AVAILABLE:
        return None
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        print(f"Could not load {fname}: {e}")
        return None

rnn_model  = load_keras("best_simplernn.keras")
lstm_model = load_keras("best_lstm.keras")
gru_model  = load_keras("best_gru.keras")

# ─────────────────────────────────────────────
# Text preprocessing (matches training pipeline)
# ─────────────────────────────────────────────
import string

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn",
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# ─────────────────────────────────────────────
# Inference functions
# ─────────────────────────────────────────────
def predict_rnn_based(text: str, model, le):
    """Run inference with SimpleRNN / LSTM / GRU."""
    if model is None or tokenizer is None or le is None:
        return {}, "Model not loaded."
    cleaned = clean_text(text)
    seq  = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    classes = le.classes_
    result = {classes[i].replace("_", " ").title(): float(probs[i]) for i in range(len(classes))}
    best   = max(result, key=result.get)
    conf   = result[best]
    return result, f"**{best}** ({conf:.1%})"


# Hugging Face Repository for DistilBERT
HF_REPO_ID = "Mohamed123AFIFI/consumer-complaint-distilbert"
DISTILBERT_AVAILABLE = True  # It will auto-download from Hugging Face

def predict_distilbert(text: str):
    """Run DistilBERT inference directly from Hugging Face Hub."""
    try:
        clf = pipeline(
            "text-classification",
            model=HF_REPO_ID,
            tokenizer=HF_REPO_ID,
            return_all_scores=True,
        )
        raw = clf(text[:512])[0]
        le  = bert_le if bert_le is not None else label_encoder
        if le is not None:
            result = {
                le.classes_[int(r["label"].split("_")[-1])].replace("_", " ").title(): r["score"]
                for r in raw
            }
        else:
            result = {r["label"]: r["score"] for r in raw}
        best = max(result, key=result.get)
        conf = result[best]
        return result, f"**{best}** ({conf:.1%})"
    except Exception as e:
        # Fallback to GRU on any error
        probs, label = predict_rnn_based(text, gru_model, label_encoder)
        return probs, label


# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────
MODEL_MAP = {
    "SimpleRNN":  lambda t: predict_rnn_based(t, rnn_model,  label_encoder),
    "LSTM":       lambda t: predict_rnn_based(t, lstm_model, label_encoder),
    "GRU":        lambda t: predict_rnn_based(t, gru_model,  label_encoder),
    "DistilBERT (Best)": predict_distilbert,
}

DESCRIPTIONS = {
    "SimpleRNN":         "Val Accuracy: 76.1% | Macro F1: 0.592",
    "LSTM":              "Val Accuracy: 88.2% | Macro F1: 0.847",
    "GRU":               "Val Accuracy: 88.4% | Macro F1: 0.849",
    "DistilBERT (Best)": "Val Accuracy: 89.7% | Macro F1: 0.865 ⭐",
}

EXAMPLE_COMPLAINTS = [
    "I noticed an unauthorized charge on my credit card statement for a subscription I never signed up for.",
    "There is an error on my credit report showing a late payment that I actually paid on time.",
    "A debt collector keeps calling me multiple times a day even after I sent a cease communication letter.",
    "My mortgage servicer applied my payment to the wrong account and is now charging me late fees.",
    "The bank froze my checking account without any prior notice and I cannot access my funds.",
]

def run_inference(complaint_text: str, model_name: str):
    if not complaint_text.strip():
        return "Please enter a complaint.", {}
    probs, prediction = MODEL_MAP[model_name](complaint_text)
    return prediction, probs

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
)

CSS = """
.gradio-container { max-width: 960px !important; margin: auto; }
.header-box { text-align: center; padding: 1.5rem 1rem 0.5rem; }
.header-box h1 { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
.header-box p  { color: #4a5568; font-size: 0.95rem; }
"""

with gr.Blocks(title="Consumer Complaint Classifier") as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box">
        <h1>🏦 Consumer Complaint Classifier</h1>
        <p>Automatically route financial consumer complaints using deep learning NLP models.<br>
           Compare SimpleRNN · LSTM · GRU · DistilBERT (fine-tuned).</p>
    </div>
    """)

    # ── Main Tab ────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("🔍 Classify Complaint"):
            with gr.Row():
                with gr.Column(scale=3):
                    complaint_input = gr.Textbox(
                        label="Customer Complaint Narrative",
                        placeholder="Enter the customer complaint text here...",
                        lines=6,
                        max_lines=12,
                    )
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="DistilBERT (Best)",
                        label="Select Model",
                    )
                    model_info = gr.Markdown(f"*{DESCRIPTIONS['DistilBERT (Best)']}*")
                    classify_btn = gr.Button("🚀 Classify", variant="primary", size="lg")

                with gr.Column(scale=2):
                    prediction_out = gr.Markdown(label="Prediction", value="*Prediction will appear here.*")
                    confidence_out = gr.Label(
                        label="Confidence per Category",
                        num_top_classes=5,
                    )

            gr.Examples(
                examples=[[c, "DistilBERT (Best)"] for c in EXAMPLE_COMPLAINTS],
                inputs=[complaint_input, model_selector],
                label="Example Complaints",
                examples_per_page=5,
            )

        with gr.Tab("📊 Model Comparison"):
            gr.Markdown("""
## Model Performance Summary

All models were trained on the **Consumer Financial Protection Bureau (CFPB)** complaint dataset,
classifying complaints into 5 product categories.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| SimpleRNN | 76.11% | 60.27% | 63.66% | 59.18% |
| LSTM | 88.21% | 84.56% | 84.88% | 84.69% |
| GRU | 88.37% | 84.70% | 85.16% | 84.91% |
| **DistilBERT** ⭐ | **89.72%** | **87.34%** | **85.66%** | **86.47%** |

### DistilBERT Per-Class Results

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Credit Card | 82.4% | 81.1% | 81.8% |
| Credit Reporting | 92.5% | 95.2% | 93.8% |
| Debt Collection | 84.4% | 79.7% | 81.9% |
| Mortgages & Loans | 88.5% | 83.0% | 85.6% |
| Retail Banking | 88.9% | 89.4% | 89.1% |

### Key Findings

- **SimpleRNN** struggled significantly (Macro F1: 59%) — its inability to capture long-range
  dependencies in complaint text limits performance on this task.
- **LSTM & GRU** performed comparably (~88%), with GRU slightly better due to fewer parameters
  and faster convergence. Both benefit from gating mechanisms for long-text memory.
- **DistilBERT** achieved the best results across all metrics by leveraging pretrained
  contextual representations. It excels especially on *Credit Reporting* (F1: 93.8%), which
  has the largest support, and *Retail Banking* (F1: 89.1%).
- **Best model for deployment: DistilBERT** (89.7% accuracy, 86.5% Macro F1).
""")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## About This Project

**Task:** Multi-class text classification of financial consumer complaints into 5 product categories.

**Dataset:** CFPB Consumer Complaint Database

**Categories:**
- 💳 Credit Card
- 📋 Credit Reporting
- 📞 Debt Collection
- 🏠 Mortgages & Loans
- 🏦 Retail Banking

**Pipeline:**
1. **EDA & Cleaning** — explore class distribution, handle imbalance
2. **Preprocessing** — lowercase, remove punctuation/numbers/stopwords, tokenize, pad
3. **SimpleRNN** — baseline sequential model
4. **LSTM** — gated memory for long sequences
5. **GRU** — efficient gating with fewer parameters
6. **DistilBERT** — HuggingFace fine-tuned transformer (best performer)
7. **Comparison** — accuracy, precision, recall, F1, confusion matrix across all models

**Tech Stack:** TensorFlow/Keras · HuggingFace Transformers · Gradio · scikit-learn
""")

    # ── Callbacks ───────────────────────────────────────────────────
    def update_model_info(model_name):
        return f"*{DESCRIPTIONS[model_name]}*"

    model_selector.change(update_model_info, inputs=model_selector, outputs=model_info)

    classify_btn.click(
        fn=run_inference,
        inputs=[complaint_input, model_selector],
        outputs=[prediction_out, confidence_out],
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=THEME, css=CSS)
