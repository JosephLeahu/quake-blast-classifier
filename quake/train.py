"""Train and evaluate earthquake-vs-blast classifiers.

    python -m quake.train                       # real data from quake.fetch
    python -m quake.train --synthetic           # offline smoke test (fake data!)

Protocol
--------
1. Chronological split: train on older events, test on the newest 20%.
2. Inside the training period, hold out the newest 20% as validation to pick
   each model's decision threshold (max F1) and to choose the best model.
3. Refit the chosen model on the full training period, evaluate once on test.
The metric that matters is PR-AUC (average precision), because blasts are a
small minority class and accuracy would be misleading.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .features import FEATURES, build_features, label_and_filter, time_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def candidate_models(seed: int = 42) -> dict:
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed,
        ),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if len(set(y_true)) > 1 else None,
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run(df_raw: pd.DataFrame, out_dir: Path = ROOT, data_label: str = "real", seed: int = 42) -> dict:
    df = label_and_filter(df_raw)
    train_df, test_df = time_split(df, 0.2)
    inner_df, val_df = time_split(train_df, 0.2)

    X_inner, y_inner = build_features(inner_df), inner_df["is_blast"].to_numpy()
    X_val, y_val = build_features(val_df), val_df["is_blast"].to_numpy()
    X_train, y_train = build_features(train_df), train_df["is_blast"].to_numpy()
    X_test, y_test = build_features(test_df), test_df["is_blast"].to_numpy()

    results, test_probas, fitted = {}, {}, {}
    for name, model in candidate_models(seed).items():
        m = clone(model).fit(X_inner, y_inner)
        val_proba = m.predict_proba(X_val)[:, 1]
        thr = best_f1_threshold(y_val, val_proba) if name != "baseline_prior" else 0.5
        val_pr_auc = float(average_precision_score(y_val, val_proba))

        final = clone(model).fit(X_train, y_train)
        proba = final.predict_proba(X_test)[:, 1]
        results[name] = {"val_pr_auc": round(val_pr_auc, 4), "test": evaluate(y_test, proba, thr)}
        test_probas[name], fitted[name] = proba, final

    best = max((n for n in results if n != "baseline_prior"), key=lambda n: results[n]["val_pr_auc"])

    # Permutation importance on (a sample of) the test set, scored by PR-AUC.
    idx = np.random.default_rng(seed).choice(len(X_test), min(len(X_test), 5000), replace=False)
    imp = permutation_importance(fitted[best], X_test.iloc[idx], y_test[idx], scoring="average_precision",
                                 n_repeats=5, random_state=seed)
    importance = sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda x: -x[1])

    summary = {
        "data": data_label,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_events": len(df),
        "blast_rate": round(float(df["is_blast"].mean()), 4),
        "train_period": [str(train_df["time"].iloc[0]), str(train_df["time"].iloc[-1])],
        "test_period": [str(test_df["time"].iloc[0]), str(test_df["time"].iloc[-1])],
        "n_train": len(train_df), "n_test": len(test_df),
        "best_model": best,
        "models": results,
        "permutation_importance": importance,
    }

    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    joblib.dump({"model": fitted[best], "threshold": results[best]["test"]["threshold"],
                 "features": FEATURES, "summary": summary}, out_dir / "models" / "model.joblib")
    (out_dir / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2))
    _plots(out_dir / "reports", y_test, test_probas, df)
    _report(out_dir / "reports" / "REPORT.md", summary)
    return summary


def _plots(out: Path, y_test: np.ndarray, probas: dict, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    ax.set_xlabel("Recall (blasts found)"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on held-out future events"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "pr_curve.png", dpi=150); plt.close(fig)

    feats = build_features(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(0, 25)
    ax.hist(feats.loc[df["is_blast"] == 0, "local_hour"], bins=bins, density=True, alpha=0.6, label="earthquake")
    ax.hist(feats.loc[df["is_blast"] == 1, "local_hour"], bins=bins, density=True, alpha=0.6, label="blast")
    ax.set_xlabel("Local hour (Pacific)"); ax.set_ylabel("Share of events"); ax.legend()
    ax.set_title("When events happen")
    fig.tight_layout(); fig.savefig(out / "hour_of_day.png", dpi=150); plt.close(fig)


def _report(path: Path, s: dict) -> None:
    rows = "\n".join(
        f"| {n} | {r['val_pr_auc']} | {r['test']['pr_auc']} | {r['test']['roc_auc']} | "
        f"{r['test']['precision']} | {r['test']['recall']} | {r['test']['f1']} |"
        for n, r in s["models"].items()
    )
    imp = "\n".join(f"| {f} | {v} |" for f, v in s["permutation_importance"][:8])
    warn = "\n> **SYNTHETIC DATA: these numbers are not real results.**\n" if s["data"] != "real" else ""
    path.write_text(f"""# Model report
{warn}
- Events: {s['n_events']:,} (blast rate {s['blast_rate']:.1%})
- Train: {s['train_period'][0]} → {s['train_period'][1]} ({s['n_train']:,})
- Test: {s['test_period'][0]} → {s['test_period'][1]} ({s['n_test']:,})
- Selected model: **{s['best_model']}** (by validation PR-AUC)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
{rows}

## Most important features (permutation, Δ PR-AUC)

| Feature | Importance |
|---|---|
{imp}

![PR curve](pr_curve.png)
![Hour of day](hour_of_day.png)
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "pnw_catalog.csv"))
    ap.add_argument("--synthetic", action="store_true", help="train on fake data (smoke test only)")
    args = ap.parse_args()

    if args.synthetic:
        from .synthetic import make_catalog
        df, label = make_catalog(6000), "synthetic"
    else:
        if not Path(args.data).exists():
            raise SystemExit(f"{args.data} not found. Run: python -m quake.fetch")
        df, label = pd.read_csv(args.data), "real"

    s = run(df, data_label=label)
    print(f"data={label} events={s['n_events']:,} blast_rate={s['blast_rate']:.1%} best={s['best_model']}")
    for name, r in s["models"].items():
        t = r["test"]
        print(f"  {name:20s} PR-AUC={t['pr_auc']:.3f}  P={t['precision']:.3f}  R={t['recall']:.3f}  F1={t['f1']:.3f}")
    print("report: reports/REPORT.md")


if __name__ == "__main__":
    main()
