"""
mlmodel.py
----------
Trains and evaluates multiple regression models to predict
the popularity score (0–100) of a Spotify track.

Models compared:
  • Random Forest Regressor
  • Gradient Boosting Regressor (XGBoost-style via sklearn)
  • Ridge Regression (baseline)

Evaluation metrics:
  • MAE  – Mean Absolute Error
  • RMSE – Root Mean Squared Error
  • R²   – Coefficient of Determination
  • ±10 accuracy – % tracks predicted within 10 popularity points
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# FEATURE DEFINITIONS
# ──────────────────────────────────────────────

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_s", "energy_dance", "acoustic_energy_ratio",
    "genre_mean_pop", "genre_std_pop"
]

CATEGORICAL_FEATURES = [
    "track_genre", "mood", "length_cat", "tempo_cat", "key", "mode", "time_signature"
]

TARGET = "popularity"


# ──────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ──────────────────────────────────────────────

def build_preprocessor(num_features, cat_features):
    """Build a ColumnTransformer that scales numeric + OHE categorical."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ])
    return preprocessor


# ──────────────────────────────────────────────
# MODEL DEFINITIONS
# ──────────────────────────────────────────────

def get_models(preprocessor):
    """Return dict of named Pipeline models."""
    models = {
        "Ridge (Baseline)": Pipeline([
            ("prep", preprocessor),
            ("model", Ridge(alpha=10.0))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ))
        ]),
    }
    return models


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Fit and evaluate a single pipeline. Returns a result dict."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    preds = np.clip(preds, 0, 100)  # popularity is bounded

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    acc10 = np.mean(np.abs(preds - y_test) <= 10) * 100  # % within ±10 pts

    print(f"\n{'─'*45}")
    print(f"  Model : {name}")
    print(f"  MAE   : {mae:.2f}")
    print(f"  RMSE  : {rmse:.2f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  ±10 acc: {acc10:.1f}%")

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "acc10": acc10,
        "pipeline": pipeline,
        "predictions": preds
    }


def cross_validate_model(name, pipeline, X, y, cv=5):
    """Run k-fold cross-validation and print mean R²."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  [{name}] CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ──────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def get_feature_importance(pipeline, num_features, cat_features, top_n=15):
    """
    Extract feature importances from a fitted tree-based pipeline.
    Returns a DataFrame sorted by importance descending.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    prep = pipeline.named_steps["prep"]
    ohe_cols = prep.named_transformers_["cat"].get_feature_names_out(cat_features)
    all_feature_names = list(num_features) + list(ohe_cols)

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": all_feature_names[:len(importances)],
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return feat_df


# ──────────────────────────────────────────────
# MAIN (standalone run)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("[mlmodel] Loading data...")
    df = pd.read_csv("data/tracks_clean.csv")

    # Drop rows with missing engineered features
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    models       = get_models(preprocessor)

    results = []
    for name, pipeline in models.items():
        res = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(res)

    print("\n\n=== SUMMARY ===")
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("pipeline", "predictions")}
        for r in results
    ])
    print(summary.to_string(index=False))
