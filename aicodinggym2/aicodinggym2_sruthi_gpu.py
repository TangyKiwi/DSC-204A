"""
Automated Essay Scoring 2.0 - GPU-optimized version
Metric: Quadratic Weighted Kappa (QWK), scores 1-6
Submit: aicodinggym mle submit learning-agency-lab-automated-essay-scoring-2 -F predictions.csv

GPU changes:
- Uses CUDA first for DistilBERT embeddings, then MPS, then CPU fallback.
- Uses torch.inference_mode(), CUDA autocast, non_blocking transfers, TF32, and larger CUDA batch size.
- Replaces sklearn HistGradientBoostingRegressor with XGBoost GPU when CUDA is available.
- Keeps TF-IDF, TruncatedSVD, Ridge, and LinearSVR on CPU because sklearn does not use CUDA.
"""

import gc
import re
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

DATA_DIR = "./learning-agency-lab-automated-essay-scoring-2/data/learning-agency-lab-automated-essay-scoring-2"
OUT_PATH = "./predictions_sruthi.csv"

# Prefer NVIDIA CUDA. MPS is okay on Apple Silicon, but XGBoost GPU below needs CUDA.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

USE_CUDA = DEVICE.type == "cuda"
print(f"Device: {DEVICE}")
if USE_CUDA:
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/ test.csv")
print(f"Train: {train.shape}  Test: {test.shape}")

# ── DistilBERT Embedding Extraction ───────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 64 if USE_CUDA else 32
PIN_MEMORY = USE_CUDA


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


def empty_device_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def get_embeddings(texts, tokenizer, model):
    ds = EssayDataset(texts, tokenizer)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=0,  # safer on Windows/local scripts; raise only inside a proper __main__ guard
    )

    model.eval()
    embeddings = []
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if USE_CUDA else nullcontext()

    with torch.inference_mode():
        for batch in dl:
            ids = batch["input_ids"].to(DEVICE, non_blocking=USE_CUDA)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=USE_CUDA)

            with amp_ctx:
                out = model(input_ids=ids, attention_mask=mask)
                hidden = out.last_hidden_state
                m = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)

            embeddings.append(pooled.detach().float().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


print("Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

print("Encoding train + test essays...")
all_texts = pd.concat([train["full_text"], test["full_text"]], ignore_index=True)
all_embeds = get_embeddings(all_texts, tokenizer, bert)

train_embeds = all_embeds[: len(train)]
test_embeds = all_embeds[len(train) :]
print(f"Embeddings shape: {train_embeds.shape}")

del bert, all_embeds
empty_device_cache()
gc.collect()

# ── Hand-Crafted Features ─────────────────────────────────────────────────────

DISCOURSE = re.compile(
    r"\b(however|therefore|furthermore|moreover|additionally|consequently|"
    r"nevertheless|although|whereas|thus|hence|in conclusion|in summary|"
    r"for example|for instance|in addition|on the other hand|as a result|"
    r"in contrast|to summarize|in other words|specifically|notably|indeed)\b",
    re.IGNORECASE,
)
LONG_WORDS = re.compile(r"\b\w{10,}\b")
VOWELS = re.compile(r"[aeiouAEIOU]")


def approx_syllables(w):
    return max(len(VOWELS.findall(w)), 1)


def text_features(df):
    texts = df["full_text"]
    words = texts.str.split()
    sents = texts.str.split(r"[.!?]+")
    paras = texts.str.split(r"\n+")

    f = pd.DataFrame(index=df.index)
    f["word_count"] = words.apply(len)
    f["char_count"] = texts.str.len()
    f["sent_count"] = sents.apply(lambda s: max(len([x for x in s if x.strip()]), 1))
    f["para_count"] = paras.apply(lambda p: max(len([x for x in p if x.strip()]), 1))
    f["unique_words"] = words.apply(lambda w: len(set(w)))
    f["vocab_richness"] = f["unique_words"] / (f["word_count"] + 1)
    f["long_word_ratio"] = texts.apply(lambda t: len(LONG_WORDS.findall(t))) / (f["word_count"] + 1)
    f["avg_word_len"] = texts.apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0)
    f["avg_sent_len"] = f["word_count"] / f["sent_count"]
    f["avg_para_len"] = f["word_count"] / f["para_count"]
    f["std_sent_len"] = sents.apply(
        lambda s: np.std([len(x.split()) for x in s if x.strip()])
        if len([x for x in s if x.strip()]) > 1
        else 0
    )
    f["comma_per_sent"] = texts.str.count(r",") / f["sent_count"]
    f["semicolon_count"] = texts.str.count(r";")
    f["discourse_ratio"] = texts.apply(lambda t: len(DISCOURSE.findall(t))) / (f["sent_count"] + 1)
    f["avg_syllables"] = texts.apply(
        lambda t: np.mean([approx_syllables(w) for w in t.split()]) if t.split() else 1
    )
    f["flesch_ease"] = 206.835 - 1.015 * f["avg_sent_len"] - 84.6 * f["avg_syllables"]
    return f.astype(np.float32)


print("Building hand features...")
train_feats = text_features(train)
test_feats = text_features(test)

# ── TF-IDF ────────────────────────────────────────────────────────────────────
# sklearn TF-IDF is CPU-only. This still matters, but the GPU gains come from BERT + XGBoost.

print("Fitting TF-IDF...")
all_raw = pd.concat([train["full_text"], test["full_text"]], ignore_index=True)
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=80_000,
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32,
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=50_000,
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32,
)
word_vec.fit(all_raw)
char_vec.fit(all_raw)


def build_sparse(df, feats):
    w = word_vec.transform(df["full_text"])
    c = char_vec.transform(df["full_text"])
    h = sp.csr_matrix(feats.values.astype(np.float32))
    return sp.hstack([w, c, h], format="csr", dtype=np.float32)


X_train_sp = build_sparse(train, train_feats)
X_test_sp = build_sparse(test, test_feats)
y_train = train["score"].values.astype(np.float32)
y_train_int = train["score"].values.astype(int)

print("Computing SVD...")
svd = TruncatedSVD(n_components=200, random_state=42)
X_train_svd = svd.fit_transform(X_train_sp).astype(np.float32)
X_test_svd = svd.transform(X_test_sp).astype(np.float32)

# Dense: BERT embeddings + SVD + hand features
X_train_dense = np.hstack([train_embeds, X_train_svd, train_feats.values]).astype(np.float32)
X_test_dense = np.hstack([test_embeds, X_test_svd, test_feats.values]).astype(np.float32)
print(f"Final dense features: {X_train_dense.shape}")

del train_embeds, test_embeds, X_train_svd, X_test_svd
empty_device_cache()
gc.collect()

# ── QWK Utilities ─────────────────────────────────────────────────────────────


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimise_thresholds(y_true, raw, n=6):
    raw = np.asarray(raw, dtype=np.float64)

    def neg_qwk(t):
        t = np.sort(t)
        p = np.clip(np.digitize(raw, t) + 1, 1, n)
        return -qwk(y_true, p)

    best_res, best_val = None, 1.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        init = np.sort(rng.uniform(raw.min(), raw.max(), n - 1))
        res = minimize(
            neg_qwk,
            init,
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-8},
        )
        if res.fun < best_val:
            best_val, best_res = res.fun, res
    return np.sort(best_res.x)


def apply_thresholds(raw, thresholds, n=6):
    return np.clip(np.digitize(raw, thresholds) + 1, 1, n).astype(int)


def make_xgb_model(fold):
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    params = dict(
        n_estimators=2500,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=0,
    )

    # XGBoost 2.x CUDA path. If your installed XGBoost is older, fallback happens in the fold loop.
    if USE_CUDA:
        params.update(tree_method="hist", device="cuda")
    else:
        params.update(tree_method="hist")

    return XGBRegressor(**params)


# ── 5-Fold CV ─────────────────────────────────────────────────────────────────

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_ridge = np.zeros(len(train), dtype=np.float32)
oof_svr = np.zeros(len(train), dtype=np.float32)
oof_xgb = np.zeros(len(train), dtype=np.float32)
pred_ridge = np.zeros(len(test), dtype=np.float32)
pred_svr = np.zeros(len(test), dtype=np.float32)
pred_xgb = np.zeros(len(test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sp, y_train_int)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")

    # CPU sparse linear model; good complement to transformer/dense features.
    ridge = Ridge(alpha=3.0, solver="lsqr")
    ridge.fit(X_train_sp[tr_idx], y_train[tr_idx])
    oof_ridge[va_idx] = ridge.predict(X_train_sp[va_idx]).astype(np.float32)
    pred_ridge += ridge.predict(X_test_sp).astype(np.float32) / N_FOLDS

    # CPU dense linear SVR. StandardScaler kept CPU because sklearn does not use CUDA.
    svr = make_pipeline(StandardScaler(), LinearSVR(C=1.0, max_iter=3000, random_state=42))
    svr.fit(X_train_dense[tr_idx], y_train[tr_idx])
    oof_svr[va_idx] = svr.predict(X_train_dense[va_idx]).astype(np.float32)
    pred_svr += svr.predict(X_test_dense).astype(np.float32) / N_FOLDS

    # GPU model when CUDA is available; replaces CPU HistGradientBoostingRegressor.
    xgb_model = make_xgb_model(fold)
    try:
        xgb_model.fit(
            X_train_dense[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train_dense[va_idx], y_train[va_idx])],
            verbose=False,
        )
    except TypeError:
        # Compatibility fallback for older XGBoost versions that do not support device="cuda".
        if USE_CUDA:
            print("  XGBoost device='cuda' unsupported; retrying with tree_method='gpu_hist'.")
            xgb_model = XGBRegressor(
                n_estimators=2500,
                learning_rate=0.025,
                max_depth=4,
                min_child_weight=2.0,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=42 + fold,
                n_jobs=-1,
                verbosity=0,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
            )
            xgb_model.fit(
                X_train_dense[tr_idx],
                y_train[tr_idx],
                eval_set=[(X_train_dense[va_idx], y_train[va_idx])],
                verbose=False,
            )
        else:
            raise

    oof_xgb[va_idx] = xgb_model.predict(X_train_dense[va_idx]).astype(np.float32)
    pred_xgb += xgb_model.predict(X_test_dense).astype(np.float32) / N_FOLDS

    for name, oof in [("Ridge", oof_ridge), ("SVR", oof_svr), ("XGB", oof_xgb)]:
        t = optimise_thresholds(y_train_int[va_idx], oof[va_idx])
        s = qwk(y_train_int[va_idx], apply_thresholds(oof[va_idx], t))
        print(f"  {name}: {s:.4f}")

    del ridge, svr, xgb_model
    empty_device_cache()
    gc.collect()

# ── Ensemble ──────────────────────────────────────────────────────────────────

print("\nSearching ensemble weights...")
best_score, best_w = -1, (0.2, 0.2, 0.6)

for w_r in np.arange(0.0, 0.7, 0.1):
    for w_s in np.arange(0.0, 0.7, 0.1):
        w_x = round(1.0 - w_r - w_s, 1)
        if w_x < 0.1:
            continue
        blend = w_r * oof_ridge + w_s * oof_svr + w_x * oof_xgb
        t = optimise_thresholds(y_train_int, blend)
        s = qwk(y_train_int, apply_thresholds(blend, t))
        if s > best_score:
            best_score, best_w = s, (w_r, w_s, w_x)

w_r, w_s, w_x = best_w
print(f"Best weights  Ridge={w_r:.1f} SVR={w_s:.1f} XGB={w_x:.1f}")
oof_blend = w_r * oof_ridge + w_s * oof_svr + w_x * oof_xgb
thresh_final = optimise_thresholds(y_train_int, oof_blend)
print(f"Final OOF QWK: {qwk(y_train_int, apply_thresholds(oof_blend, thresh_final)):.4f}")

# ── predictions.csv ───────────────────────────────────────────────────────────

pred_blend = w_r * pred_ridge + w_s * pred_svr + w_x * pred_xgb
final_preds = apply_thresholds(pred_blend, thresh_final)

out = pd.DataFrame({"essay_id": test["essay_id"], "score": final_preds})
out.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH}")
print(out["score"].value_counts().sort_index().to_string())
