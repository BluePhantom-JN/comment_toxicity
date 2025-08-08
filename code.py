# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# Ensure nltk resources (first run)
# ---------------------------
nltk_packages = ['stopwords','wordnet','punkt']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except:
        nltk.download(pkg)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Text cleaning / preprocessing
# ---------------------------
def clean_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if (w.isalpha()) and (w not in stop_words)]
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens if len(w) >= 3]
    tokens = [w for w in tokens if len(w) >= 3]
    return tokens

# ---------------------------
# Model classes (same as in your notebook)
# ---------------------------
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        x = self.fc1(hidden.squeeze(0))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextLSTM, self).__init__()
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
        x = torch.relu(self.fc1(hidden.squeeze(0)))
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# ---------------------------
# Utility: build / load word2idx
# ---------------------------
@st.cache_data(show_spinner=False)
def build_word2idx_from_df(df, min_freq=1):
    word2idx = {'<pad>':0, '<unk>':1}
    idx = 2
    freq = {}
    for tokens in df['tokens']:
        for w in tokens:
            freq[w] = freq.get(w,0)+1
    for w,c in freq.items():
        if c >= min_freq:
            word2idx[w] = idx
            idx += 1
    return word2idx

def tokens_to_tensor(tokens, word2idx):
    idxs = [word2idx.get(w, word2idx.get('<unk>',1)) for w in tokens]
    if len(idxs) == 0:
        return torch.tensor([0], dtype=torch.long)  # single pad
    return torch.tensor(idxs, dtype=torch.long)

def prepare_batch_from_texts(texts, word2idx, device='cpu'):
    vectors = [tokens_to_tensor(clean_text(t), word2idx) for t in texts]
    pad_seq = pad_sequence(vectors, batch_first=True, padding_value=0).to(device)
    return pad_seq

# ---------------------------
# Load dataset (train.csv) if available for insights & building vocab
# ---------------------------
DATA_CSV = 'train.csv'  # change if different
df = None
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'comment_text' in df.columns:
        df['tokens'] = df['comment_text'].apply(clean_text)
    else:
        # minimal fallback
        df['tokens'] = df.iloc[:,0].astype(str).apply(clean_text)

# ---------------------------
# Load or build word2idx (try saved pickle first)
# ---------------------------
WORD2IDX_PKL = 'word2idx.pkl'
if os.path.exists(WORD2IDX_PKL):
    with open(WORD2IDX_PKL,'rb') as f:
        word2idx = pickle.load(f)
else:
    if df is not None:
        word2idx = build_word2idx_from_df(df)
        # save for future runs
        with open(WORD2IDX_PKL,'wb') as f:
            pickle.dump(word2idx,f)
    else:
        # fallback very small default
        word2idx = {'<pad>':0,'<unk>':1}

# ---------------------------
# Load models
# ---------------------------
DEVICE = torch.device('cpu')

# RNN: expects state_dict file (rnn_model_state_dict.pth)
RNN_FILE = 'rnn_model_state_dict.pth'
LSTM_FILE = 'model.pth'  # saved entire model in your notebook

rnn_model = None
lstm_model = None

# We need vocab size for model init:
vocab_size = len(word2idx)
EMBED_DIM = 64
HIDDEN_DIM = 128
OUT_DIM = 6
try:
    if os.path.exists(RNN_FILE):
        rnn_model = TextRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, OUT_DIM).to(DEVICE)
        state = torch.load(RNN_FILE, map_location=DEVICE)
        # If saved as state_dict:
        if isinstance(state, dict) and not isinstance(state.get('__class__',None),str):
            rnn_model.load_state_dict(state)
        rnn_model.eval()
except Exception as e:
    st.warning(f"Could not load RNN model: {e}")

try:
    if os.path.exists(LSTM_FILE):
        # model.pth in your notebook was saved as `torch.save(model1, 'model.pth')`
        # load with map_location
        maybe = torch.load(LSTM_FILE, map_location=DEVICE)
        if isinstance(maybe, nn.Module):
            lstm_model = maybe
            # Make sure embedding layers match vocab size; if not, re-init and load state if it's a state_dict inside
            if hasattr(lstm_model, 'embedding') and lstm_model.embedding.num_embeddings != vocab_size:
                # try to adjust by creating a new model and copying states for common indices
                temp = TextLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, OUT_DIM)
                try:
                    temp_state = lstm_model.state_dict()
                    temp.load_state_dict(temp_state, strict=False)
                except:
                    pass
                lstm_model = temp
        elif isinstance(maybe, dict):
            # If a state dict was saved instead
            temp = TextLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, OUT_DIM)
            temp.load_state_dict(maybe)
            lstm_model = temp
        lstm_model.to(DEVICE)
        lstm_model.eval()
except Exception as e:
    st.warning(f"Could not load LSTM model: {e}")

# ---------------------------
# Labels
# ---------------------------
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# ---------------------------
# Prediction helpers
# ---------------------------
@st.cache_data
def predict_batch(texts, which='lstm', threshold=0.5):
    model = lstm_model if which == 'lstm' else rnn_model
    if model is None:
        raise RuntimeError(f"{which} model not loaded.")
    inp = prepare_batch_from_texts(texts, word2idx, device=DEVICE)
    with torch.no_grad():
        start = time.time()
        out = model(inp)
        elapsed = time.time() - start
        probs = out.cpu().numpy()
        preds = (probs >= threshold).astype(int)
    return probs, preds, elapsed

def predict_single(text, which='lstm', threshold=0.5):
    p, pred, elapsed = predict_batch([text], which, threshold)
    return p[0], pred[0], elapsed

# ---------------------------
# Compute metrics on a sample (if dataset available)
# ---------------------------
@st.cache_data
def compute_metrics_on_df(which='lstm', threshold=0.5, sample_n=2000):
    if df is None:
        return None
    # take sample for speed
    use_df = df.copy()
    # ensure required label columns exist:
    for col in LABELS:
        if col not in use_df.columns:
            use_df[col] = 0
    if len(use_df) > sample_n:
        use_df = use_df.sample(sample_n, random_state=42)
    texts = use_df['comment_text'].astype(str).tolist() if 'comment_text' in use_df.columns else use_df.iloc[:,0].astype(str).tolist()
    probs, preds, elapsed = predict_batch(texts, which=which, threshold=threshold)
    true = use_df[LABELS].values
    # per label metrics
    per_label = []
    for i, lab in enumerate(LABELS):
        p = precision_score(true[:,i], preds[:,i], zero_division=0)
        r = recall_score(true[:,i], preds[:,i], zero_division=0)
        f1 = f1_score(true[:,i], preds[:,i], zero_division=0)
        per_label.append({'label':lab,'precision':p,'recall':r,'f1':f1})
    overall = {
        'accuracy': accuracy_score(true.flatten(), preds.flatten()),
        'precision_micro': precision_score(true, preds, average='micro', zero_division=0),
        'recall_macro': recall_score(true, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(true, preds, average='macro', zero_division=0),
        'avg_inference_time_sec': elapsed / max(1, len(texts))
    }
    return per_label, overall

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.set_page_config(layout='wide', page_title='Toxicity Detector — RNN vs LSTM')
st.title("Toxicity Detection — **Compare RNN & LSTM**")
st.markdown("Enter a comment for live prediction, or upload a CSV for bulk predictions.\
             The app will show **ETA**, **per-label metrics**, and **model evolution** if available.")

# Sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.01)
models_choice = st.sidebar.multiselect("Models to use / compare", ['lstm','rnn'], default=['lstm','rnn'])
sample_n = st.sidebar.number_input("Sample size for computing metrics (when dataset available)", min_value=200, max_value=10000, value=2000, step=100)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Bulk CSV", "Insights & Metrics", "Model Evolution / Loss"])

# ---------------------------
# Tab 1: Real-time prediction
# ---------------------------
with tab1:
    st.header("Single comment prediction (real-time)")
    text = st.text_area("Enter comment here", height=150, placeholder="Type or paste a comment...")
    col1, col2 = st.columns(2)
    if st.button("Predict"):
        if not text or text.strip()=="":
            st.warning("Please enter a comment.")
        else:
            results = {}
            for which in models_choice:
                if (which == 'lstm' and lstm_model is None) or (which == 'rnn' and rnn_model is None):
                    results[which] = {"error":"model not loaded"}
                    continue
                probs, preds, elapsed = predict_batch([text], which=which, threshold=threshold)
                probs = probs[0]
                preds = preds[0]
                # compose
                df_out = pd.DataFrame({
                    'label': LABELS,
                    'probability': probs,
                    'prediction': preds
                })
                results[which] = {
                    'table': df_out,
                    'inference_time_sec': elapsed
                }
            # Display
            for which, info in results.items():
                st.subheader(which.upper())
                if 'error' in info:
                    st.error(info['error'])
                    continue
                st.write(f"Inference time: **{info['inference_time_sec']*1000:.1f} ms**")
                st.dataframe(info['table'].assign(probability=lambda d: d['probability'].map(lambda v: f"{v:.3f}")))
            # show combined quick verdict
            if all('table' in v for v in results.values()):
                combined = pd.concat([results[w]['table'].set_index('label')['prediction'].rename(f'pred_{w}') for w in results], axis=1)
                st.subheader("Compare model predictions (1 = predicted positive)")
                st.dataframe(combined)

# ---------------------------
# Tab 2: Bulk CSV upload
# ---------------------------
with tab2:
    st.header("Bulk predictions — upload CSV")
    st.markdown("CSV should contain a column named `comment_text` (or the first column will be used).")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        uploaded_df = pd.read_csv(uploaded)
        # choose column
        if 'comment_text' in uploaded_df.columns:
            texts = uploaded_df['comment_text'].astype(str).tolist()
        else:
            texts = uploaded_df.iloc[:,0].astype(str).tolist()
        st.write(f"Number of rows: {len(texts)}")
        # run predictions for each selected model and append columns
        out_df = uploaded_df.copy()
        timing_summary = {}
        for which in models_choice:
            if (which == 'lstm' and lstm_model is None) or (which == 'rnn' and rnn_model is None):
                st.warning(f"{which} not loaded; skipping.")
                continue
            t0 = time.time()
            probs, preds, elapsed = predict_batch(texts, which=which, threshold=threshold)
            t1 = time.time()
            timing_summary[which] = {'batch_time_sec': t1 - t0, 'per_sample_sec': (t1-t0)/max(1,len(texts))}
            # attach probabilities and predictions as columns (one col per label)
            for i, lab in enumerate(LABELS):
                out_df[f'{which}_prob_{lab}'] = probs[:,i]
                out_df[f'{which}_pred_{lab}'] = preds[:,i]
        st.write("Inference timing:")
        st.json(timing_summary)
        st.write("Preview of predictions:")
        st.dataframe(out_df.head(20))
        # download button
        csv_buffer = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv_buffer, file_name="predictions.csv", mime='text/csv')

# ---------------------------
# Tab 3: Insights & Metrics
# ---------------------------
with tab3:
    st.header("Dataset insights & model metrics")
    if df is None:
        st.info("train.csv not found — put your dataset file named 'train.csv' in app folder to see data insights and compute metrics.")
    else:
        st.subheader("Data insights")
        # class distribution
        cols = [c for c in LABELS if c in df.columns]
        cd = df[cols].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        cd.plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Number of positive comments')
        ax1.set_title('Label distribution')
        st.pyplot(fig1)
        # token length distribution
        df['len'] = df['tokens'].str.len()
        fig2, ax2 = plt.subplots()
        sns.histplot(df['len'], bins=50, kde=True, ax=ax2)
        ax2.set_title('Token length distribution')
        st.pyplot(fig2)

    st.subheader("Model comparison — per-label metrics (computed on sample)")
    if df is None:
        st.info("No dataset to compute metrics. Upload train.csv to enable this.")
    else:
        results_summary = {}
        for which in models_choice:
            if (which == 'lstm' and lstm_model is None) or (which == 'rnn' and rnn_model is None):
                st.warning(f"{which} not loaded — skipping metrics for it.")
                continue
            per_label, overall = compute_metrics_on_df(which=which, threshold=threshold, sample_n=int(sample_n))
            pl_df = pd.DataFrame(per_label)
            st.markdown(f"**{which.upper()}** per-label metrics (threshold={threshold})")
            st.dataframe(pl_df.style.format({'precision':'{:.3f}','recall':'{:.3f}','f1':'{:.3f}'}))
            st.markdown("Overall")
            st.write(overall)
            results_summary[which] = {'per_label':pl_df, 'overall':overall}
        # If both available, show side-by-side table
        if len(results_summary) == 2:
            left, right = st.columns(2)
            w1, w2 = list(results_summary.keys())
            left.subheader(w1.upper() + " overall")
            left.write(results_summary[w1]['overall'])
            right.subheader(w2.upper() + " overall")
            right.write(results_summary[w2]['overall'])

# ---------------------------
# Tab 4: Model evolution / training loss
# ---------------------------
with tab4:
    st.header("Model evolution / training loss plots (if saved)")
    found = False
    for fname in ['loss_df.csv','loss_df1.csv','loss_df.pkl','loss_df1.pkl','loss_df','loss_df1']:
        if os.path.exists(fname):
            found = True
            try:
                if fname.endswith('.csv'):
                    ld = pd.read_csv(fname)
                elif fname.endswith('.pkl'):
                    with open(fname,'rb') as f:
                        ld = pickle.load(f)
                else:
                    continue
                st.markdown(f"### Loss history from `{fname}`")
                st.dataframe(ld.head())
                fig, ax = plt.subplots()
                if 'Epoch' in ld.columns and 'Loss' in ld.columns:
                    sns.lineplot(data=ld, x='Epoch', y='Loss', ax=ax)
                elif 'epoch' in ld.columns and 'loss' in ld.columns:
                    sns.lineplot(data=ld, x='epoch', y='loss', ax=ax)
                else:
                    # try generic numeric column plot
                    numeric_cols = ld.select_dtypes(include='number').columns.tolist()
                    if len(numeric_cols) >= 1:
                        sns.lineplot(data=ld[numeric_cols], ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
    if not found:
        st.info("No training loss history files (loss_df*). If you have them, save them as CSV or PKL in app folder with names like 'loss_df.csv', 'loss_df1.csv'.")

# ---------------------------
# Optional: sample testcases per label (shows examples and model predictions)
# ---------------------------
st.markdown("---")
st.subheader("Sample testcases per label (from train.csv if available)")
if df is None:
    st.info("No train.csv found.")
else:
    for lab in LABELS:
        if lab in df.columns:
            subset = df[df[lab] == 1]
            if len(subset) > 0:
                example = subset.sample(1).iloc[0]
                txt = example['comment_text'] if 'comment_text' in df.columns else ' '.join(example['tokens'])
                st.markdown(f"**{lab}** example:")
                st.write(txt)
                # Show model predictions on this sample
                for which in ['lstm','rnn']:
                    if (which == 'lstm' and lstm_model is None) or (which == 'rnn' and rnn_model is None):
                        continue
                    probs, preds, elapsed = predict_batch([txt], which=which, threshold=threshold)
                    st.write(f"{which.upper()} -> probs: {np.round(probs[0],3)} preds: {preds[0]} time: {elapsed*1000:.1f} ms")
            else:
                st.write(f"No examples for {lab}")

st.caption("App created from user's notebook. Ensure `train.csv`, `rnn_model_state_dict.pth`, and `model.pth` are present in the same folder for full functionality.")
