"""
experiment.py — Extended experiment for the Algorithm Strategies paper.

Compares Levenshtein Distance, Jaro-Winkler, and Jaccard Similarity
for prompt-injection detection using sliding-window fuzzy matching
with nested 5-fold cross-validation.

Outputs:
  - CSV files with per-fold metrics, confusion matrices, score distributions
  - PNG visualisations (bar chart, confusion matrices, box plots, line chart, timing)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from rapidfuzz.distance import Levenshtein, JaroWinkler

# configuration
OUT_DIR   = os.path.join(os.path.dirname(__file__), "images")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

METHODS   = ["lev", "jw", "jac"]
METHOD_LABELS = {
    "lev": "Levenshtein",
    "jw":  "Jaro-Winkler",
    "jac": "Jaccard"
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

COLORS = {
    "lev": "#2563EB",   # blue
    "jw":  "#16A34A",   # green
    "jac": "#DC2626",   # red
}

# dataset
print("Loading dataset ...")
ds = load_dataset("deepset/prompt-injections")
df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
X  = df["text"].to_numpy(dtype=str)
y  = df["label"].to_numpy(dtype=int)
print(f"Dataset loaded: {len(X)} samples  (injection={y.sum()}, benign={len(y)-y.sum()})")

# helper funcs
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def jaccard(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if (A | B) else 0.0

def best_score(text, pattern, method):
    words = text.split()
    p_len = len(pattern.split())
    best  = 0.0
    for w in range(max(2, p_len - 2), p_len + 3):
        if len(words) < w:
            continue
        for i in range(len(words) - w + 1):
            chunk = " ".join(words[i:i + w])
            if method == "lev":
                s = Levenshtein.normalized_similarity(chunk, pattern)
            elif method == "jw":
                s = JaroWinkler.similarity(chunk, pattern)
            else:
                s = jaccard(chunk, pattern)
            if s > best:
                best = s
    return best

def doc_score(text, patterns, method):
    text = preprocess(text)
    return max((best_score(text, p, method) for p in patterns), default=0.0)

def build_patterns(texts, labels):
    return [preprocess(t) for t, l in zip(texts, labels) if l == 1]

def tune_threshold(scores, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.50, 0.95, 0.01):
        pred = (scores >= t).astype(int)
        f1   = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return best_t

# storage
fold_metrics   = {m: [] for m in METHODS}   # list of dicts per fold
agg_cm         = {m: np.zeros((2, 2), dtype=int) for m in METHODS}
score_dists    = {m: {"injection": [], "benign": []} for m in METHODS}
exec_times     = {m: [] for m in METHODS}   # seconds per fold
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# fold loop
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n{'='*20} FOLD {fold} {'='*20}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_pat, X_val, y_pat, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    patterns = build_patterns(X_pat, y_pat)

    val_scores = {}
    for m in METHODS:
        val_scores[m] = np.array([doc_score(t, patterns, m) for t in X_val])

    thresholds = {m: tune_threshold(val_scores[m], y_val) for m in METHODS}
    print(f"  Thresholds -> Lev: {thresholds['lev']:.2f} | "
          f"JW: {thresholds['jw']:.2f} | Jac: {thresholds['jac']:.2f}")

    for m in METHODS:
        t0 = time.perf_counter()
        test_scores = np.array([doc_score(t, patterns, m) for t in X_test])
        elapsed = time.perf_counter() - t0
        exec_times[m].append(elapsed)

        preds = (test_scores >= thresholds[m]).astype(int)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        cm   = confusion_matrix(y_test, preds, labels=[0, 1])

        fold_metrics[m].append({
            "fold": fold,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "threshold": round(thresholds[m], 2),
            "time_sec": round(elapsed, 2),
        })
        agg_cm[m] += cm

        score_dists[m]["injection"].extend(test_scores[y_test == 1].tolist())
        score_dists[m]["benign"].extend(test_scores[y_test == 0].tolist())

        print(f"  {METHOD_LABELS[m]:14s}  Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}  t={elapsed:.1f}s")

# save csv files
all_rows = []
for m in METHODS:
    for d in fold_metrics[m]:
        all_rows.append({"method": METHOD_LABELS[m], **d})
metrics_df = pd.DataFrame(all_rows)
metrics_df.to_csv(os.path.join(DATA_DIR, "fold_metrics.csv"), index=False)

# confusion matrices
for m in METHODS:
    pd.DataFrame(agg_cm[m], index=["Actual 0","Actual 1"],
                 columns=["Pred 0","Pred 1"]).to_csv(
        os.path.join(DATA_DIR, f"cm_{m}.csv"))

# score distributions
for m in METHODS:
    pd.DataFrame({
        "score": score_dists[m]["injection"] + score_dists[m]["benign"],
        "label": (["injection"]*len(score_dists[m]["injection"]) +
                  ["benign"]*len(score_dists[m]["benign"]))
    }).to_csv(os.path.join(DATA_DIR, f"scores_{m}.csv"), index=False)

print("\n[V] CSV files saved to", DATA_DIR)

# 1. Bar chart: average metrics comparison ──
fig, ax = plt.subplots(figsize=(6, 3.8))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_keys  = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_names))
w = 0.22

for i, m in enumerate(METHODS):
    vals = [np.mean([d[k] for d in fold_metrics[m]]) for k in metric_keys]
    bars = ax.bar(x + i*w, vals, w, label=METHOD_LABELS[m], color=COLORS[m],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + w)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Average Evaluation Metrics Comparison (5-Fold CV)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "metrics_comparison.png"))
plt.close(fig)
print("[V] metrics_comparison.png")

# 2. Confusion matrix heatmaps
for m in METHODS:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    cm = agg_cm[m]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Injection"])
    ax.set_yticklabels(["Benign", "Injection"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {METHOD_LABELS[m]}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{m}.png"))
    plt.close(fig)
    print(f"[V] confusion_matrix_{m}.png")

# 3. Box plot: score distributions
fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=True)
for ax, m in zip(axes, METHODS):
    inj = score_dists[m]["injection"]
    ben = score_dists[m]["benign"]
    bp = ax.boxplot([ben, inj], labels=["Benign", "Injection"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    bp["boxes"][0].set_facecolor("#93C5FD")
    bp["boxes"][1].set_facecolor("#FCA5A5")
    ax.set_title(METHOD_LABELS[m])
    ax.set_ylabel("Similarity Score" if m == "lev" else "")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Similarity Score Distribution per Class", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "score_distribution.png"))
plt.close(fig)
print("[V] score_distribution.png")

# 4. Line chart: F1 per fold
fig, ax = plt.subplots(figsize=(5, 3.5))
folds = list(range(1, 6))
for m in METHODS:
    f1s = [d["f1"] for d in fold_metrics[m]]
    ax.plot(folds, f1s, marker="o", label=METHOD_LABELS[m],
            color=COLORS[m], linewidth=1.8, markersize=5)
ax.set_xticks(folds)
ax.set_xlabel("Fold")
ax.set_ylabel("F1-Score")
ax.set_title("F1-Score per Fold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "f1_per_fold.png"))
plt.close(fig)
print("[V] f1_per_fold.png")

# 5. Bar chart: average execution time
fig, ax = plt.subplots(figsize=(4.5, 3.2))
avg_times = [np.mean(exec_times[m]) for m in METHODS]
bars = ax.bar([METHOD_LABELS[m] for m in METHODS], avg_times,
              color=[COLORS[m] for m in METHODS],
              edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Time (seconds)")
ax.set_title("Average Execution Time per Fold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "execution_time.png"))
plt.close(fig)
print("[V] execution_time.png")

# summary
print("\n" + "="*55)
print("FINAL AVERAGE (5-FOLD CV)")
print("="*55)
for m in METHODS:
    vals = fold_metrics[m]
    print(f"  {METHOD_LABELS[m]:14s}  "
          f"Acc={np.mean([d['accuracy'] for d in vals]):.4f}  "
          f"Prec={np.mean([d['precision'] for d in vals]):.4f}  "
          f"Rec={np.mean([d['recall'] for d in vals]):.4f}  "
          f"F1={np.mean([d['f1'] for d in vals]):.4f}  "
          f"Time={np.mean(exec_times[m]):.1f}s")

# Save summary JSON
summary = {}
for m in METHODS:
    vals = fold_metrics[m]
    summary[m] = {
        "accuracy":  round(np.mean([d["accuracy"] for d in vals]), 4),
        "precision": round(np.mean([d["precision"] for d in vals]), 4),
        "recall":    round(np.mean([d["recall"] for d in vals]), 4),
        "f1":        round(np.mean([d["f1"] for d in vals]), 4),
        "time_avg":  round(np.mean(exec_times[m]), 2),
        "folds":     vals,
    }
with open(os.path.join(DATA_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[V] All outputs saved to {OUT_DIR} and {DATA_DIR}")