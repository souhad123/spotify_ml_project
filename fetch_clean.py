"""
fetch_clean.py
--------------
Fetches and cleans the Spotify Tracks dataset from Kaggle.
Uses the `spotipy` library for optional live API enrichment,
and a static CSV fallback for reproducibility.

Dataset: Spotify Tracks Dataset (Kaggle)
URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

def load_data(filepath: str = "data/tracks.csv") -> pd.DataFrame:
    """Load the raw Spotify tracks CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download it from Kaggle:\n"
            "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset\n"
            "and place it in the data/ directory."
        )
    df = pd.read_csv(filepath, index_col=0)
    print(f"[load]  Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# 2. BASIC CLEANING
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicates, handle missing values, fix dtypes.
    Returns a cleaned DataFrame.
    """
    initial_rows = len(df)

    # Drop unnamed / useless columns
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["track_id"])

    # Drop rows with missing critical fields
    critical_cols = ["track_name", "artists", "popularity", "danceability",
                     "energy", "loudness", "tempo", "duration_ms"]
    df = df.dropna(subset=critical_cols)

    # Remove tracks with popularity == 0 and duration < 30s (likely previews/errors)
    df = df[df["popularity"] > 0]
    df = df[df["duration_ms"] >= 30_000]

    # Fix duration: convert ms → seconds
    df["duration_s"] = df["duration_ms"] / 1000.0
    df = df.drop(columns=["duration_ms"])

    # Clamp loudness (some extreme outliers)
    df["loudness"] = df["loudness"].clip(lower=-60, upper=5)

    print(f"[clean] {initial_rows:,} → {len(df):,} rows after cleaning "
          f"(removed {initial_rows - len(df):,})")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-informed features derived from audio attributes.
    """
    # Energy × Danceability interaction
    df["energy_dance"] = df["energy"] * df["danceability"]

    # Acoustic mismatch: high energy + high acousticness = unusual
    df["acoustic_energy_ratio"] = df["acousticness"] / (df["energy"] + 1e-6)

    # Valence bucket: negative / neutral / positive mood
    df["mood"] = pd.cut(
        df["valence"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["negative", "neutral", "positive"]
    )

    # Duration bucket: short / standard / long
    df["length_cat"] = pd.cut(
        df["duration_s"],
        bins=[0, 150, 270, 99_999],
        labels=["short", "standard", "long"]
    )

    # Speechiness flag: tracks heavy on spoken words
    df["is_speech_heavy"] = (df["speechiness"] > 0.33).astype(int)

    # Tempo bucket: slow / mid / fast
    df["tempo_cat"] = pd.cut(
        df["tempo"],
        bins=[0, 90, 140, 9999],
        labels=["slow", "mid", "fast"]
    )

    print(f"[feat]  Feature engineering done → {df.shape[1]} columns total")
    return df


# ──────────────────────────────────────────────
# 4. GENRE AGGREGATION
# ──────────────────────────────────────────────

def aggregate_genre_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-genre mean popularity and merge back as a feature.
    This captures the 'genre premium' a track benefits from.
    """
    genre_stats = (
        df.groupby("track_genre")["popularity"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "genre_mean_pop", "std": "genre_std_pop"})
        .reset_index()
    )
    df = df.merge(genre_stats, on="track_genre", how="left")
    print("[feat]  Genre-level aggregation merged.")
    return df


# ──────────────────────────────────────────────
# 5. SAVE PROCESSED DATA
# ──────────────────────────────────────────────

def save_processed(df: pd.DataFrame, out_path: str = "data/tracks_clean.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[save]  Saved cleaned dataset → {out_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data/tracks.csv")
    df = clean_data(df)
    df = engineer_features(df)
    df = aggregate_genre_stats(df)
    save_processed(df, "data/tracks_clean.csv")
    print("\nSample of processed data:")
    print(df[["track_name", "artists", "popularity", "energy_dance", "mood"]].head(5))
