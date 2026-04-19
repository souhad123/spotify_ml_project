"""
main.py
-------
Entry point: orchestrates the full Spotify Popularity ML pipeline.

Usage:
    python main.py

Steps:
  1. Load & clean data  (fetch_clean.py)
  2. Train & evaluate models  (mlmodel.py)
  3. Plot results & feature importances
  4. Print final summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

from fetch_clean import load_data, clean_data, engineer_features, aggregate_genre_stats
from mlmodel import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
    build_preprocessor, get_models, evaluate_model, get_feature_importance
)

os.makedirs("outputs", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#1DB954", "#191414", "#535353"]   # Spotify green / dark / grey


# ──────────────────────────────────────────────
# STEP 1 – DATA PIPELINE
# ──────────────────────────────────────────────

def run_data_pipeline():
    print("=" * 55)
    print("  STEP 1 — Data Pipeline")
    print("=" * 55)
    df = load_data("data/tracks.csv")
    df = clean_data(df)
    df = engineer_features(df)
    df = aggregate_genre_stats(df)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    print(f"\n  Final dataset: {len(df):,} tracks | {df.shape[1]} features\n")
    return df


# ──────────────────────────────────────────────
# STEP 2 – MODEL TRAINING
# ──────────────────────────────────────────────

def run_models(df):
    print("=" * 55)
    print("  STEP 2 — Model Training & Evaluation")
    print("=" * 55)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    models       = get_models(preprocessor)

    results = []
    for name, pipeline in models.items():
        res = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        res["y_test"] = y_test.values
        results.append(res)

    return results, X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# STEP 3 – VISUALIZATIONS
# ──────────────────────────────────────────────

def plot_model_comparison(results):
    """Bar chart comparing MAE, RMSE, R² across models."""
    names   = [r["model"] for r in results]
    mae_vals  = [r["mae"]  for r in results]
    rmse_vals = [r["rmse"] for r in results]
    r2_vals   = [r["r2"]   for r in results]
    acc_vals  = [r["acc10"] for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Model Comparison — Spotify Popularity Prediction", fontsize=14, fontweight="bold")

    for ax, vals, label, color, invert in zip(
        axes,
        [mae_vals, rmse_vals, r2_vals, acc_vals],
        ["MAE ↓ (lower is better)", "RMSE ↓ (lower is better)",
         "R² ↑ (higher is better)", "±10 Accuracy % ↑"],
        PALETTE + ["#E87722"],
        [True, True, False, False]
    ):
        bars = ax.bar(names, vals, color=color, edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        if invert:
            ax.set_ylim(0, max(vals) * 1.25)

    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.close()
    print("\n  [plot] Saved: outputs/model_comparison.png")


def plot_predictions(results):
    """Scatter: predicted vs actual popularity for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Predicted vs Actual Popularity", fontsize=14, fontweight="bold")

    for ax, res, color in zip(axes, results, PALETTE):
        y_true = res["y_test"]
        y_pred = res["predictions"]
        ax.scatter(y_true, y_pred, alpha=0.15, s=8, color=color)
        lims = [0, 100]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual Popularity", fontsize=10)
        ax.set_ylabel("Predicted Popularity", fontsize=10)
        ax.set_title(f"{res['model']}\nMAE={res['mae']:.1f} | R²={res['r2']:.3f}", fontsize=10)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/predictions_scatter.png", dpi=150)
    plt.close()
    print("  [plot] Saved: outputs/predictions_scatter.png")


def plot_feature_importance(results):
    """Horizontal bar of top-15 features for the best tree model."""
    best = next((r for r in results if "Random Forest" in r["model"]), results[-1])
    feat_df = get_feature_importance(
        best["pipeline"], NUMERIC_FEATURES, CATEGORICAL_FEATURES, top_n=15
    )
    if feat_df is None:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1],
            color=PALETTE[0], edgecolor="white")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("  [plot] Saved: outputs/feature_importance.png")


def plot_popularity_distribution(df):
    """Distribution of track popularity in the dataset."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["popularity"], bins=50, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.axvline(df["popularity"].median(), color="red", linestyle="--",
               label=f"Median = {df['popularity'].median():.0f}")
    ax.set_title("Distribution of Track Popularity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Popularity (0–100)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/popularity_distribution.png", dpi=150)
    plt.close()
    print("  [plot] Saved: outputs/popularity_distribution.png")


# ──────────────────────────────────────────────
# STEP 4 – SUMMARY
# ──────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("pipeline", "predictions", "y_test")}
        for r in results
    ])
    print(summary.to_string(index=False))

    best = summary.loc[summary["mae"].idxmin()]
    print(f"\n  ✓ Best model by MAE : {best['model']}")
    print(f"    MAE  = {best['mae']:.2f}")
    print(f"    RMSE = {best['rmse']:.2f}")
    print(f"    R²   = {best['r2']:.4f}")
    print(f"    ±10 acc = {best['acc10']:.1f}%")
    print("\n  Outputs saved in ./outputs/")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎵  Spotify Popularity Prediction — ML Pipeline\n")

    df                              = run_data_pipeline()
    results, X_train, X_test, y_train, y_test = run_models(df)

    print("\n" + "=" * 55)
    print("  STEP 3 — Plots")
    print("=" * 55)
    plot_popularity_distribution(df)
    plot_model_comparison(results)
    plot_predictions(results)
    plot_feature_importance(results)

    print_summary(results)
