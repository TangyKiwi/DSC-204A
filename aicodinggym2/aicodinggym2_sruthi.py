"""
Automated Essay Scoring 2.0
Metric: Quadratic Weighted Kappa (QWK), scores 1-6
Submit: aicodinggym mle submit learning-agency-lab-automated-essay-scoring-2 -F predictions.csv
"""

import re
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data" / "learning-agency-lab-automated-essay-scoring-2"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
print(f"Train: {train.shape}  Test: {test.shape}")

# ── BERT Embedding Extraction ─────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 512
BATCH_SIZE = 32

class EssayDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.enc = tokenizer(
            texts.tolist(),
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    def __len__(self):
        return self.enc["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


def get_embeddings(texts, tokenizer, model):
    ds = EssayDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            # mean-pool over non-padding tokens
            hidden = out.last_hidden_state
            m      = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(1) / m.sum(1)
            embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(embeddings)


print("Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert      = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

print("Encoding train essays...")
all_texts  = pd.concat([train["full_text"], test["full_text"]], ignore_index=True)
all_embeds = get_embeddings(all_texts, tokenizer, bert)

train_embeds = all_embeds[:len(train)]
test_embeds  = all_embeds[len(train):]
print(f"Embeddings shape: {train_embeds.shape}")

del bert
if DEVICE.type == "mps":
    torch.mps.empty_cache()

# ── Hand-Crafted Features ─────────────────────────────────────────────────────

DISCOURSE = re.compile(
    r"\b(however|therefore|furthermore|moreover|additionally|consequently|"
    r"nevertheless|although|whereas|thus|hence|in conclusion|in summary|"
    r"for example|for instance|in addition|on the other hand|as a result|"
    r"in contrast|to summarize|in other words|specifically|notably|indeed)\b",
    re.IGNORECASE,
)
LONG_WORDS = re.compile(r"\b\w{10,}\b")


def text_features(df):
    texts = df["full_text"]
    words  = texts.str.split()
    sents  = texts.str.split(r"[.!?]+")
    paras  = texts.str.split(r"\n+")

    f = pd.DataFrame(index=df.index)
    f["word_count"]      = words.apply(len)
    f["char_count"]      = texts.str.len()
    f["sent_count"]      = sents.apply(lambda s: max(len([x for x in s if x.strip()]), 1))
    f["para_count"]      = paras.apply(lambda p: max(len([x for x in p if x.strip()]), 1))
    f["unique_words"]    = words.apply(lambda w: len(set(w)))
    f["vocab_richness"]  = f["unique_words"] / (f["word_count"] + 1)
    f["long_word_ratio"] = texts.apply(lambda t: len(LONG_WORDS.findall(t))) / (f["word_count"] + 1)
    f["avg_word_len"]    = texts.apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0)
    f["avg_sent_len"]    = f["word_count"] / f["sent_count"]
    f["avg_para_len"]    = f["word_count"] / f["para_count"]
    f["std_sent_len"]    = sents.apply(
        lambda s: np.std([len(x.split()) for x in s if x.strip()]) if len([x for x in s if x.strip()]) > 1 else 0
    )
    f["comma_per_sent"]  = texts.str.count(r",") / f["sent_count"]
    f["semicolon_count"] = texts.str.count(r";")
    f["discourse_ratio"] = texts.apply(lambda t: len(DISCOURSE.findall(t))) / (f["sent_count"] + 1)
    def approx_syllables(w): return max(len(re.findall(r"[aeiouAEIOU]", w)), 1)
    f["avg_syllables"]   = texts.apply(lambda t: np.mean([approx_syllables(w) for w in t.split()]) if t.split() else 1)
    f["flesch_ease"]     = 206.835 - 1.015 * f["avg_sent_len"] - 84.6 * f["avg_syllables"]
    return f.astype(np.float32)


print("Building hand features...")
train_feats = text_features(train)
test_feats  = text_features(test)

# ── TF-IDF ────────────────────────────────────────────────────────────────────

print("Fitting TF-IDF...")
all_raw = pd.concat([train["full_text"], test["full_text"]])
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=80_000, min_df=2, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=50_000, min_df=3, sublinear_tf=True)
word_vec.fit(all_raw)
char_vec.fit(all_raw)


def build_sparse(df, feats):
    w = word_vec.transform(df["full_text"])
    c = char_vec.transform(df["full_text"])
    h = sp.csr_matrix(feats.values)
    return sp.hstack([w, c, h], format="csr")


X_train_sp = build_sparse(train, train_feats)
X_test_sp  = build_sparse(test,  test_feats)
y_train    = train["score"].values

print("Computing SVD...")
svd = TruncatedSVD(n_components=200, random_state=42)
X_train_svd = svd.fit_transform(X_train_sp)
X_test_svd  = svd.transform(X_test_sp)

# Dense: BERT embeddings + SVD + hand features
X_train_dense = np.hstack([train_embeds, X_train_svd, train_feats.values])
X_test_dense  = np.hstack([test_embeds,  X_test_svd,  test_feats.values])
print(f"Final dense features: {X_train_dense.shape}")

# ── QWK Utilities ─────────────────────────────────────────────────────────────

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimise_thresholds(y_true, raw, n=6):
    def neg_qwk(t):
        t = np.sort(t)
        p = np.clip(np.digitize(raw, t) + 1, 1, n)
        return -qwk(y_true, p)
    # multiple restarts to avoid local optima
    best_res, best_val = None, 1.0
    for seed in range(5):
        rng  = np.random.default_rng(seed)
        init = np.sort(rng.uniform(raw.min(), raw.max(), n - 1))
        res  = minimize(neg_qwk, init, method="Nelder-Mead",
                        options={"maxiter": 20_000, "xatol": 1e-8})
        if res.fun < best_val:
            best_val, best_res = res.fun, res
    return np.sort(best_res.x)


def apply_thresholds(raw, thresholds, n=6):
    return np.clip(np.digitize(raw, thresholds) + 1, 1, n)

# ── 5-Fold CV ─────────────────────────────────────────────────────────────────

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_ridge  = np.zeros(len(train))
oof_svr    = np.zeros(len(train))
oof_hgbr   = np.zeros(len(train))
pred_ridge = np.zeros(len(test))
pred_svr   = np.zeros(len(test))
pred_hgbr  = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sp, y_train)):
    print(f"\nFold {fold+1}/{N_FOLDS}")

    ridge = Ridge(alpha=3.0, solver="lsqr")
    ridge.fit(X_train_sp[tr_idx], y_train[tr_idx])
    oof_ridge[va_idx]  = ridge.predict(X_train_sp[va_idx])
    pred_ridge        += ridge.predict(X_test_sp) / N_FOLDS

    svr = make_pipeline(StandardScaler(), LinearSVR(C=1.0, max_iter=2000))
    svr.fit(X_train_dense[tr_idx], y_train[tr_idx])
    oof_svr[va_idx]  = svr.predict(X_train_dense[va_idx])
    pred_svr        += svr.predict(X_test_dense) / N_FOLDS

    hgbr = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1,
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30,
    )
    hgbr.fit(X_train_dense[tr_idx], y_train[tr_idx])
    oof_hgbr[va_idx]  = hgbr.predict(X_train_dense[va_idx])
    pred_hgbr        += hgbr.predict(X_test_dense) / N_FOLDS

    for name, oof in [("Ridge", oof_ridge), ("SVR", oof_svr), ("HGBR", oof_hgbr)]:
        t = optimise_thresholds(y_train[va_idx], oof[va_idx])
        s = qwk(y_train[va_idx], apply_thresholds(oof[va_idx], t))
        print(f"  {name}: {s:.4f}")

# ── Ensemble ──────────────────────────────────────────────────────────────────

print("\nSearching ensemble weights...")
best_score, best_w = -1, (0.2, 0.3, 0.5)

for w_r in np.arange(0.1, 0.6, 0.1):
    for w_s in np.arange(0.1, 0.6, 0.1):
        w_h = round(1.0 - w_r - w_s, 1)
        if w_h < 0.1:
            continue
        blend = w_r * oof_ridge + w_s * oof_svr + w_h * oof_hgbr
        t     = optimise_thresholds(y_train, blend)
        s     = qwk(y_train, apply_thresholds(blend, t))
        if s > best_score:
            best_score, best_w = s, (w_r, w_s, w_h)

w_r, w_s, w_h = best_w
print(f"Best weights  Ridge={w_r:.1f} SVR={w_s:.1f} HGBR={w_h:.1f}")
oof_blend    = w_r * oof_ridge    + w_s * oof_svr    + w_h * oof_hgbr
thresh_final = optimise_thresholds(y_train, oof_blend)
print(f"Final OOF QWK: {qwk(y_train, apply_thresholds(oof_blend, thresh_final)):.4f}")

# ── predictions.csv ───────────────────────────────────────────────────────────

pred_blend  = w_r * pred_ridge + w_s * pred_svr + w_h * pred_hgbr
final_preds = apply_thresholds(pred_blend, thresh_final)

out = pd.DataFrame({"essay_id": test["essay_id"], "score": final_preds})
out.to_csv(Path(__file__).parent / "predictions.csv", index=False)
print("\nSaved predictions.csv")
print(out["score"].value_counts().sort_index().to_string())