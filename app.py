# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter

# -----------------------
# Ensure NLTK resources
# -----------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

nltk_packages = ["wordnet", "punkt"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except Exception:
        try:
            nltk.download(pkg)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()

# -----------------------
# Config / constants
# -----------------------
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DEFAULT_MAX_LEN = 8            # matches your filtering (len<=8)
THRESHOLD = 0.39               # you used 0.39 in evaluation

# -----------------------
# Helper functions
# -----------------------
def clean_text(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [w for w in tokens if (w.isalpha()) and (w not in stop_words)]
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens if len(w) >= 3]
    tokens = [w for w in tokens if len(w) >= 3]
    return tokens

def text_to_tensor(tokens, word2idx, max_len=DEFAULT_MAX_LEN):
    ids = [word2idx.get(w, 1) for w in tokens]
    if len(ids) >= max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # batch_size=1
    return tensor

# -----------------------
# Define model classes
# -----------------------
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=6):
        super().__init__()
        self.embeding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embeding(x)
        output, hidden = self.rnn(x)
        x = self.fc1(hidden.squeeze(0))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = F.relu(self.fc1(hidden.squeeze(0)))
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# -----------------------
# Caching model & resources
# -----------------------
@st.cache_resource
def load_word2idx_from_bytes(b: bytes):
    return pickle.loads(b)

@st.cache_resource
def load_word2idx_from_fileobj(fobj):
    return pickle.load(fobj)

@st.cache_resource
def prepare_model_from_state_dict_bytes(state_bytes, model_type, vocab_size):
    try:
        state = torch.load(io.BytesIO(state_bytes), map_location='cpu')
    except Exception:
        state = None

    if model_type == 'rnn':
        model = TextRNN(vocab_size)
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state
    else:
        model = TextLSTM(vocab_size)
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state
    model.eval()
    return model

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Toxicity Detection Dashboard", layout="wide")
st.title("🛡 Toxicity Detection — Streamlit")

# Sidebar: upload model files / word2idx
st.sidebar.header("Model & Vocab uploads (optional)")
word2idx_file = st.sidebar.file_uploader("Upload word2idx.pkl (pickled dict)", type=["pkl","pickle"], key="w2i")
rnn_file = st.sidebar.file_uploader("Upload rnn_model_state_dict.pth (state_dict)", type=["pth","pt"], key="rnn")
lstm_file = st.sidebar.file_uploader("Upload model_fp16.pth / model.pth", type=["pth","pt"], key="lstm")
model_choice = st.sidebar.selectbox("Prefer model type (if both present)", ["auto","rnn","lstm"])

# Load word2idx
word2idx = None
if word2idx_file:
    try:
        word2idx = load_word2idx_from_fileobj(word2idx_file)
    except Exception:
        try:
            word2idx_file.seek(0)
            word2idx = load_word2idx_from_bytes(word2idx_file.read())
        except Exception as e:
            st.sidebar.error(f"Failed loading word2idx: {e}")

if word2idx is None:
    st.sidebar.warning("No word2idx uploaded — predictions will use a tiny fallback vocabulary.")
    word2idx = {'<pad>':0,'<unk>':1}

vocab_size = max(word2idx.values()) + 1

# Load model
loaded_model = None
loaded_model_name = None
if (rnn_file is not None) and (model_choice in ("auto","rnn")):
    try:
        rnn_bytes = rnn_file.read()
        loaded_model = prepare_model_from_state_dict_bytes(rnn_bytes, model_type='rnn', vocab_size=vocab_size)
        loaded_model_name = 'rnn'
        st.sidebar.success("RNN model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load RNN: {e}")

if (loaded_model is None) and (lstm_file is not None) and (model_choice in ("auto","lstm")):
    try:
        lstm_bytes = lstm_file.read()
        loaded_model = prepare_model_from_state_dict_bytes(lstm_bytes, model_type='lstm', vocab_size=vocab_size)
        loaded_model_name = 'lstm'
        st.sidebar.success("LSTM model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load LSTM: {e}")

if loaded_model is None:
    st.sidebar.info("No model uploaded. App will still run but predictions are disabled.")

# Prediction helper
def predict_single_text(text, model, word2idx, max_len=DEFAULT_MAX_LEN, threshold=THRESHOLD):
    tokens = clean_text(text)
    tensor = text_to_tensor(tokens, word2idx, max_len=max_len)
    if model is None:
        return {"error": "no_model", "tokens": tokens}
    with torch.no_grad():
        out = model(tensor)
        probs = out.squeeze(0).cpu().numpy().astype(float)
        binary = (probs > threshold).astype(int)
        res = {label: {"prob": float(probs[i]), "pred": int(binary[i])} for i,label in enumerate(LABELS)}
        return {"tokens": tokens, "probs": probs.tolist(), "binary": binary.tolist(), "per_label": res}

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Prediction for Text",
    "2️⃣ Bulk CSV File",
    "3️⃣ Insights",
    "4️⃣ Model Evolution Scores"
])

# --- Tab 1: Single prediction ---
with tab1:
    st.header("Single Comment Prediction")
    input_text = st.text_area("Comment", height=140, placeholder="Type or paste a comment here...")
    col1, col2 = st.columns([1,1])
    with col1:
        max_len_input = st.number_input("Max length (tokens)", min_value=1, max_value=64, value=DEFAULT_MAX_LEN)
    with col2:
        threshold_input = st.number_input("Threshold", min_value=0.01, max_value=0.99, value=THRESHOLD, format="%.2f")

    if st.button("Predict", key="single_predict"):
        if not input_text.strip():
            st.warning("Please enter a comment")
        else:
            out = predict_single_text(input_text, loaded_model, word2idx, max_len=max_len_input, threshold=threshold_input)
            if out.get("error") == "no_model":
                st.error("No model loaded. Please upload a model in the sidebar.")
            else:
                st.subheader("Tokens")
                st.write(out["tokens"])
                st.subheader("Predictions")
                df_out = pd.DataFrame({
                    "label": LABELS,
                    "probability": [round(p,4) for p in out["probs"]],
                    "predicted": out["binary"]
                })
                st.dataframe(df_out)
                positives = df_out[df_out["predicted"]==1]
                if len(positives):
                    st.warning(f"Labels predicted positive: {', '.join(positives['label'].tolist())}")
                else:
                    st.success("No toxic labels predicted")

# --- Tab 2: Bulk CSV ---
with tab2:
    st.header("Bulk prediction from CSV")
    uploaded = st.file_uploader("Upload CSV with column `comment_text`", type=["csv"], key="bulk")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None and 'comment_text' in df.columns:
            run_preview = st.checkbox("Show preview (first 10 rows)", value=True)
            if st.button("Run bulk predictions", key="bulk_run"):
                results = []
                for text in df['comment_text'].astype(str).fillna(''):
                    out = predict_single_text(text, loaded_model, word2idx, max_len=DEFAULT_MAX_LEN, threshold=THRESHOLD)
                    if out.get("error") == "no_model":
                        st.error("No model loaded. Please upload one.")
                        results = None
                        break
                    probs, binary = out["probs"], out["binary"]
                    row = {}
                    for i,label in enumerate(LABELS):
                        row[f"{label}_prob"] = probs[i]
                        row[f"{label}_pred"] = binary[i]
                    results.append(row)
                if results is not None:
                    results_df = pd.DataFrame(results)
                    final_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                    st.dataframe(final_df.head(20) if run_preview else final_df)
                    csv_bytes = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions CSV", csv_bytes, "predictions.csv", "text/csv")

# --- Tab 3: Insights ---
with tab3:
    st.header("Data Insights")
    try:
        df = pd.read_csv("insights.csv")
        lab = LABELS
        st.subheader("Label Distribution (Pie Chart)")
        fig1, ax1 = plt.subplots()
        df[lab].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        st.subheader("Label Counts (Bar Chart)")
        fig2, ax2 = plt.subplots()
        df[lab].sum().sort_values(ascending=False).plot(kind='bar', ax=ax2)
        ax2.set_ylabel("Number of Comments")
        st.pyplot(fig2)

        df['label_sum'] = df[lab].sum(axis=1)
        st.subheader("Label Sum Distribution")
        fig3, ax3 = plt.subplots()
        df['label_sum'].value_counts().sort_index().plot(kind='bar', ax=ax3)
        ax3.set_xlabel("Number of Toxic Categories")
        st.pyplot(fig3)

        if 'len' in df.columns:
            st.subheader("Length Stats (Non-toxic)")
            st.write(df[df['label_sum'] == 0]['len'].describe())
            st.subheader("Length Stats (Toxic)")
            st.write(df[df['label_sum'] > 0]['len'].describe())
            st.subheader("Histogram of Comment Lengths")
            fig4, ax4 = plt.subplots()
            sns.histplot(df['len'], bins=50, kde=True, ax=ax4)
            st.pyplot(fig4)
    except Exception as e:
        st.warning(f"No insights.csv found or error: {e}")

# --- Tab 4: Metrics ---
with tab4:
    st.header("Model evolution & scores")
    metrics_file = st.file_uploader("Upload metrics CSV", type=["csv"], key="metrics")
    if metrics_file is not None:
        try:
            df_metrics = pd.read_csv(metrics_file)
            st.dataframe(df_metrics.head(20))
            numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                cols_to_plot = st.multiselect("Choose metrics to plot", numeric_cols, default=numeric_cols[:2])
                if cols_to_plot:
                    fig5, ax5 = plt.subplots(figsize=(8,4))
                    for col in cols_to_plot:
                        ax5.plot(df_metrics.index, df_metrics[col], marker='o', label=col)
                    ax5.legend()
                    st.pyplot(fig5)
            csv_b = df_metrics.to_csv(index=False).encode('utf-8')
            st.download_button("Download metrics CSV", csv_b, "metrics.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed to read metrics: {e}")

# Footer
st.markdown("---")
st.markdown("**Tip:** Upload `word2idx.pkl` and your saved model (`rnn_model_state_dict.pth` or `model_fp16.pth`) for reproducible predictions.")
