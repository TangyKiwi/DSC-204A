"""
Automated Essay Scoring 2.0 - GPU ensemble v2
Metric: Quadratic Weighted Kappa (QWK), scores 1-6
Submit: aicodinggym mle submit learning-agency-lab-automated-essay-scoring-2 -F predictions_sruthi_v2.csv

Main v2 changes vs previous GPU file:
- Fixes the accidental " test.csv" path typo.
- Uses CUDA first, then MPS, then CPU.
- Keeps sparse TF-IDF Ridge as a strong CPU baseline.
- Adds richer transformer pooling: mean + CLS + max pooled embeddings.
- Adds optional stronger transformer model via AES_MODEL_NAME environment variable.
- Adds TWO GPU XGBoost models:
    1) regressor: predicts continuous score.
    2) classifier: predicts class probabilities, then converts to expected score.
- Adds repeated CV seeds for a more stable leaderboard prediction.
- Uses finer ensemble-weight search instead of 0.1 steps only.

Recommended first run:
    python aicodinggym2_sruthi_gpu_v2.py

Stronger but slower run if your GPU has enough VRAM and the model is available:
    AES_MODEL_NAME=microsoft/deberta-v3-small python aicodinggym2_sruthi_gpu_v2.py
"""

import gc
import os
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
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get(
    "AES_DATA_DIR",
    "./learning-agency-lab-automated-essay-scoring-2/data/learning-agency-lab-automated-essay-scoring-2",
))
OUT_PATH = Path(os.environ.get("AES_OUT_PATH", "./predictions_sruthi_v2.csv"))

# DistilBERT is safe/fast. DeBERTa v3 small often improves embeddings but is slower.
MODEL_NAME = os.environ.get("AES_MODEL_NAME", "distilbert-base-uncased")
MAX_LEN = int(os.environ.get("AES_MAX_LEN", "512"))
N_FOLDS = int(os.environ.get("AES_N_FOLDS", "5"))
CV_SEEDS = [int(x) for x in os.environ.get("AES_CV_SEEDS", "42,2024").split(",") if x.strip()]
SVD_COMPONENTS = int(os.environ.get("AES_SVD_COMPONENTS", "256"))

# Larger batch for CUDA; reduce if VRAM is tight.

# Prefer NVIDIA CUDA. MPS is useful for embedding extraction only; XGBoost GPU needs CUDA.
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
BATCH_SIZE = int(os.environ.get("AES_BATCH_SIZE", "48" if USE_CUDA else "24"))
PIN_MEMORY = USE_CUDA

print(f"Device: {DEVICE}")
if USE_CUDA:
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformer: {MODEL_NAME}")
print(f"CV seeds: {CV_SEEDS}")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"Train: {train.shape}  Test: {test.shape}")

# -----------------------------------------------------------------------------
# Transformer embedding extraction
# -----------------------------------------------------------------------------

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


def pool_hidden_states(hidden, mask):
    """Return mean + CLS + max pooled representations."""
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)

    mean_pool = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
    cls_pool = hidden[:, 0]

    # Max-pool only real tokens, not padding.
    neg_inf = torch.finfo(hidden.dtype).min
    max_hidden = hidden.masked_fill(mask_f == 0, neg_inf)
    max_pool = max_hidden.max(dim=1).values

    return torch.cat([mean_pool, cls_pool, max_pool], dim=1)


def get_embeddings(texts, tokenizer, model):
    ds = EssayDataset(texts, tokenizer)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=0,
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
                pooled = pool_hidden_states(out.last_hidden_state, mask)

            embeddings.append(pooled.detach().float().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


print("Loading transformer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

print("Encoding train + test essays...")
all_texts = pd.concat([train["full_text"], test["full_text"]], ignore_index=True)
all_embeds = get_embeddings(all_texts, tokenizer, bert)
train_embeds = all_embeds[: len(train)]
test_embeds = all_embeds[len(train):]
print(f"Transformer embeddings shape: {train_embeds.shape}")

del bert, all_embeds
empty_device_cache()
gc.collect()

# -----------------------------------------------------------------------------
# Hand-crafted features
# -----------------------------------------------------------------------------

DISCOURSE = re.compile(
    r"\b(however|therefore|furthermore|moreover|additionally|consequently|"
    r"nevertheless|although|whereas|thus|hence|in conclusion|in summary|"
    r"for example|for instance|in addition|on the other hand|as a result|"
    r"in contrast|to summarize|in other words|specifically|notably|indeed)\b",
    re.IGNORECASE,
)
LONG_WORDS = re.compile(r"\b\w{10,}\b")
VOWELS = re.compile(r"[aeiouAEIOU]")
WORD_RE = re.compile(r"\b\w+\b")


def approx_syllables(w):
    return max(len(VOWELS.findall(w)), 1)


def text_features(df):
    texts = df["full_text"].fillna("")
    lower = texts.str.lower()
    words = lower.str.findall(WORD_RE)
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
    f["avg_word_len"] = words.apply(lambda ws: np.mean([len(w) for w in ws]) if ws else 0)
    f["avg_sent_len"] = f["word_count"] / f["sent_count"]
    f["avg_para_len"] = f["word_count"] / f["para_count"]
    f["std_sent_len"] = sents.apply(
        lambda s: np.std([len(x.split()) for x in s if x.strip()])
        if len([x for x in s if x.strip()]) > 1
        else 0
    )
    f["comma_per_sent"] = texts.str.count(r",") / f["sent_count"]
    f["semicolon_count"] = texts.str.count(r";")
    f["question_count"] = texts.str.count(r"\?")
    f["exclamation_count"] = texts.str.count(r"!")
    f["quote_count"] = texts.str.count(r"[\"']")
    f["uppercase_ratio"] = texts.apply(lambda t: sum(ch.isupper() for ch in t) / (len(t) + 1))
    f["digit_ratio"] = texts.str.count(r"\d") / (f["char_count"] + 1)
    f["discourse_ratio"] = texts.apply(lambda t: len(DISCOURSE.findall(t))) / (f["sent_count"] + 1)
    f["avg_syllables"] = words.apply(lambda ws: np.mean([approx_syllables(w) for w in ws]) if ws else 1)
    f["flesch_ease"] = 206.835 - 1.015 * f["avg_sent_len"] - 84.6 * f["avg_syllables"]
    f["words_per_char"] = f["word_count"] / (f["char_count"] + 1)
    f["unique_per_sent"] = f["unique_words"] / (f["sent_count"] + 1)
    return f.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)


print("Building hand features...")
train_feats = text_features(train)
test_feats = text_features(test)
print(f"Hand features: {train_feats.shape[1]}")

# -----------------------------------------------------------------------------
# TF-IDF and SVD
# -----------------------------------------------------------------------------

print("Fitting TF-IDF...")
all_raw = pd.concat([train["full_text"], test["full_text"]], ignore_index=True).fillna("")
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=120_000,
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32,
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 6),
    max_features=80_000,
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32,
)
word_vec.fit(all_raw)
char_vec.fit(all_raw)


def build_sparse(df, feats):
    text = df["full_text"].fillna("")
    w = word_vec.transform(text)
    c = char_vec.transform(text)
    h = sp.csr_matrix(feats.values.astype(np.float32))
    return sp.hstack([w, c, h], format="csr", dtype=np.float32)


X_train_sp = build_sparse(train, train_feats)
X_test_sp = build_sparse(test, test_feats)
y_train = train["score"].values.astype(np.float32)
y_train_int = train["score"].values.astype(int)
y_train_cls = y_train_int - 1

print("Computing SVD...")
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
X_train_svd = svd.fit_transform(X_train_sp).astype(np.float32)
X_test_svd = svd.transform(X_test_sp).astype(np.float32)

# Dense: transformer embeddings + SVD + hand features.
X_train_dense = np.hstack([train_embeds, X_train_svd, train_feats.values]).astype(np.float32)
X_test_dense = np.hstack([test_embeds, X_test_svd, test_feats.values]).astype(np.float32)
print(f"Final dense features: {X_train_dense.shape}")

del train_embeds, test_embeds, X_train_svd, X_test_svd
empty_device_cache()
gc.collect()

# -----------------------------------------------------------------------------
# QWK utilities
# -----------------------------------------------------------------------------


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def apply_thresholds(raw, thresholds, n=6):
    return np.clip(np.digitize(raw, thresholds) + 1, 1, n).astype(int)


def optimise_thresholds(y_true, raw, n=6):
    raw = np.asarray(raw, dtype=np.float64)

    def neg_qwk(t):
        t = np.sort(t)
        return -qwk(y_true, apply_thresholds(raw, t, n=n))

    # Stable score-based initial cuts plus random restarts.
    base_inits = [
        np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        np.percentile(raw, [8, 25, 55, 82, 94]),
        np.linspace(raw.min(), raw.max(), n + 1)[1:-1],
    ]

    best_res, best_val = None, 1.0
    for init in base_inits:
        res = minimize(
            neg_qwk,
            np.sort(init),
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-8},
        )
        if res.fun < best_val:
            best_val, best_res = res.fun, res

    for seed in range(8):
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


def expected_score_from_proba(proba):
    classes = np.arange(1, 7, dtype=np.float32)
    return np.dot(proba.astype(np.float32), classes)


def xgb_common_params(fold_seed):
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    params = dict(
        n_estimators=3500,
        learning_rate=0.018,
        max_depth=3,
        min_child_weight=2.0,
        subsample=0.88,
        colsample_bytree=0.80,
        reg_alpha=0.08,
        reg_lambda=2.0,
        random_state=fold_seed,
        n_jobs=-1,
        verbosity=0,
    )
    if USE_CUDA:
        params.update(tree_method="hist", device="cuda")
    else:
        params.update(tree_method="hist")
    return params


def make_xgb_regressor(fold_seed):
    params = xgb_common_params(fold_seed)
    params.update(objective="reg:squarederror", eval_metric="rmse")
    return XGBRegressor(**params)


def make_xgb_classifier(fold_seed):
    params = xgb_common_params(fold_seed)
    params.update(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=6,
        # Classifier benefits from a little more depth less often than regressor.
        max_depth=4,
        reg_lambda=2.5,
    )
    return XGBClassifier(**params)


def fit_xgb_with_fallback(model, X_tr, y_tr, X_va, y_va):
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model
    except TypeError:
        if not USE_CUDA:
            raise
        # Older XGBoost fallback.
        params = model.get_params()
        params.pop("device", None)
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        model.__class__(**params)
        new_model = model.__class__(**params)
        new_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return new_model

# -----------------------------------------------------------------------------
# Repeated CV training
# -----------------------------------------------------------------------------

n_train = len(train)
n_test = len(test)

# Average OOF/preds across repeated folds.
oof_ridge = np.zeros(n_train, dtype=np.float32)
oof_svr = np.zeros(n_train, dtype=np.float32)
oof_xgb_reg = np.zeros(n_train, dtype=np.float32)
oof_xgb_cls = np.zeros(n_train, dtype=np.float32)

pred_ridge = np.zeros(n_test, dtype=np.float32)
pred_svr = np.zeros(n_test, dtype=np.float32)
pred_xgb_reg = np.zeros(n_test, dtype=np.float32)
pred_xgb_cls = np.zeros(n_test, dtype=np.float32)

oof_counts = np.zeros(n_train, dtype=np.float32)

for seed in CV_SEEDS:
    print(f"\n========== CV seed {seed} ==========")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sp, y_train_int)):
        fold_seed = seed + fold * 100
        print(f"\nFold {fold + 1}/{N_FOLDS} seed={seed}")

        ridge = Ridge(alpha=2.0, solver="lsqr")
        ridge.fit(X_train_sp[tr_idx], y_train[tr_idx])
        oof_ridge[va_idx] += ridge.predict(X_train_sp[va_idx]).astype(np.float32)
        pred_ridge += ridge.predict(X_test_sp).astype(np.float32) / (N_FOLDS * len(CV_SEEDS))

        svr = make_pipeline(StandardScaler(), LinearSVR(C=0.8, max_iter=5000, random_state=fold_seed))
        svr.fit(X_train_dense[tr_idx], y_train[tr_idx])
        oof_svr[va_idx] += svr.predict(X_train_dense[va_idx]).astype(np.float32)
        pred_svr += svr.predict(X_test_dense).astype(np.float32) / (N_FOLDS * len(CV_SEEDS))

        xgb_reg = make_xgb_regressor(fold_seed)
        xgb_reg = fit_xgb_with_fallback(
            xgb_reg,
            X_train_dense[tr_idx],
            y_train[tr_idx],
            X_train_dense[va_idx],
            y_train[va_idx],
        )
        oof_xgb_reg[va_idx] += xgb_reg.predict(X_train_dense[va_idx]).astype(np.float32)
        pred_xgb_reg += xgb_reg.predict(X_test_dense).astype(np.float32) / (N_FOLDS * len(CV_SEEDS))

        xgb_cls = make_xgb_classifier(fold_seed + 17)
        xgb_cls = fit_xgb_with_fallback(
            xgb_cls,
            X_train_dense[tr_idx],
            y_train_cls[tr_idx],
            X_train_dense[va_idx],
            y_train_cls[va_idx],
        )
        va_proba = xgb_cls.predict_proba(X_train_dense[va_idx])
        te_proba = xgb_cls.predict_proba(X_test_dense)
        oof_xgb_cls[va_idx] += expected_score_from_proba(va_proba)
        pred_xgb_cls += expected_score_from_proba(te_proba) / (N_FOLDS * len(CV_SEEDS))

        oof_counts[va_idx] += 1.0

        # Print fold-level single-model checks after averaging this fold's assigned prediction.
        for name, arr in [
            ("Ridge", oof_ridge),
            ("SVR", oof_svr),
            ("XGB-reg", oof_xgb_reg),
            ("XGB-cls", oof_xgb_cls),
        ]:
            fold_raw = arr[va_idx] / oof_counts[va_idx]
            t = optimise_thresholds(y_train_int[va_idx], fold_raw)
            s = qwk(y_train_int[va_idx], apply_thresholds(fold_raw, t))
            print(f"  {name}: {s:.4f}")

        del ridge, svr, xgb_reg, xgb_cls
        empty_device_cache()
        gc.collect()

# Finish averaging OOF predictions.
oof_ridge /= oof_counts
oof_svr /= oof_counts
oof_xgb_reg /= oof_counts
oof_xgb_cls /= oof_counts

# -----------------------------------------------------------------------------
# Ensemble search
# -----------------------------------------------------------------------------

single_models = {
    "ridge": oof_ridge,
    "svr": oof_svr,
    "xgb_reg": oof_xgb_reg,
    "xgb_cls": oof_xgb_cls,
}
print("\nSingle model OOF scores:")
for name, raw in single_models.items():
    t = optimise_thresholds(y_train_int, raw)
    s = qwk(y_train_int, apply_thresholds(raw, t))
    print(f"  {name}: {s:.5f}")

print("\nSearching ensemble weights...")
best_score = -1.0
best_w = None
best_thresh = None

# Finer but still cheap grid. XGB models usually carry the ensemble, so allow wider range.
grid = np.arange(0.0, 1.01, 0.05)
for w_r in grid:
    for w_s in grid:
        for w_g in grid:
            w_c = round(1.0 - w_r - w_s - w_g, 2)
            if w_c < 0 or w_c > 1:
                continue
            if w_g + w_c < 0.25:
                continue
            blend = (
                w_r * oof_ridge
                + w_s * oof_svr
                + w_g * oof_xgb_reg
                + w_c * oof_xgb_cls
            )
            t = optimise_thresholds(y_train_int, blend)
            s = qwk(y_train_int, apply_thresholds(blend, t))
            if s > best_score:
                best_score = s
                best_w = (w_r, w_s, w_g, w_c)
                best_thresh = t

w_r, w_s, w_g, w_c = best_w
print(
    f"Best weights: Ridge={w_r:.2f} SVR={w_s:.2f} "
    f"XGB-reg={w_g:.2f} XGB-cls={w_c:.2f}"
)
print(f"Final OOF QWK: {best_score:.5f}")
print(f"Thresholds: {np.round(best_thresh, 4)}")

# -----------------------------------------------------------------------------
# predictions.csv
# -----------------------------------------------------------------------------

pred_blend = w_r * pred_ridge + w_s * pred_svr + w_g * pred_xgb_reg + w_c * pred_xgb_cls
final_preds = apply_thresholds(pred_blend, best_thresh)

out = pd.DataFrame({"essay_id": test["essay_id"], "score": final_preds})
out.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH}")
print(out["score"].value_counts().sort_index().to_string())
