"""Prepare the support-ticket dataset for classifier training (M6, FR-1).

Cleans backend/data/raw/hf_support_tickets_main.csv, keeps only English rows,
normalizes labels through app.ml.features, and writes stratified 70/15/15 splits to
backend/data/processed/{train,val,test}.csv (random_state=42, stratified on the joint
category|severity key so both label distributions stay stable in every split).

With --download, fetches the dataset CSV from Hugging Face's public endpoint when the
raw file is absent — see Docs/DATA.md for provenance and license (cc-by-nc-4.0:
attribution required, non-commercial use only).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.features import (
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    normalize_category,
    normalize_severity,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DEFAULT = BACKEND_DIR / "data" / "raw" / "hf_support_tickets_main.csv"
PROCESSED_DEFAULT = BACKEND_DIR / "data" / "processed"
SEED = 42

# https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets (see Docs/DATA.md)
DOWNLOAD_URL = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets"
    "/resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)
RAW_FILENAME = "hf_support_tickets_main.csv"

TEXT_COLUMNS = {"subject": "subject", "body": "body"}
LABEL_COLUMNS = {"type": "category", "priority": "severity"}
LANGUAGE_COLUMN = "language"
KEEP_LANGUAGE = "en"  # dataset is multilingual (EN/DE); the classifier serves EN tickets


def download_raw(raw_path: Path) -> None:
    """Fetch the dataset CSV from Hugging Face's public endpoint."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading dataset from {DOWNLOAD_URL} ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / RAW_FILENAME
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=300) as response, tmp_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        shutil.copyfile(tmp_path, raw_path)
    print(f"saved raw CSV to {raw_path}")


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """Read the raw CSV, keep EN rows, drop nulls in required columns, normalize labels."""
    frame = pd.read_csv(raw_path)
    before = len(frame)
    required = [*TEXT_COLUMNS, *LABEL_COLUMNS]
    if LANGUAGE_COLUMN in frame.columns:
        frame = frame[frame[LANGUAGE_COLUMN] == KEEP_LANGUAGE]
    en_rows = len(frame)
    frame = frame.dropna(subset=required)
    cleaned = pd.DataFrame(
        {
            "subject": frame["subject"].astype(str),
            "body": frame["body"].astype(str),
            "category": [normalize_category(v) for v in frame["type"]],
            "severity": [normalize_severity(v) for v in frame["priority"]],
        }
    )
    print(f"loaded {before} rows from {raw_path}")
    print(f"kept {en_rows} '{KEEP_LANGUAGE}' row(s); dropped {before - en_rows} other-language row(s)")
    print(f"dropped {en_rows - len(cleaned)} row(s) with nulls in required columns")
    return cleaned


def stratified_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 split, stratified on the joint category|severity key (seed 42)."""
    strata = frame["category"] + "|" + frame["severity"]
    train_df, hold_df = train_test_split(frame, test_size=0.30, random_state=SEED, stratify=strata)
    hold_strata = hold_df["category"] + "|" + hold_df["severity"]
    val_df, test_df = train_test_split(
        hold_df, test_size=0.50, random_state=SEED, stratify=hold_strata
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_distribution_table(splits: dict[str, pd.DataFrame], column: str, labels: tuple[str, ...]) -> None:
    header = f"{'label':<24}" + "".join(f"{name:>8}" for name in splits)
    print(f"\n{column} distribution:\n{header}")
    for label in labels:
        row = f"{label:<24}"
        row += "".join(f"{int((splits[name][column] == label).sum()):>8}" for name in splits)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT, help="path to the raw dataset CSV")
    parser.add_argument(
        "--out-dir", type=Path, default=PROCESSED_DEFAULT, help="directory for processed splits"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch the dataset CSV via Hugging Face's public endpoint if the raw file is absent",
    )
    args = parser.parse_args()

    raw_path = args.raw
    if not raw_path.is_file():
        if not args.download:
            print(f"raw CSV not found at {raw_path}; re-run with --download to fetch it", file=sys.stderr)
            sys.exit(1)
        download_raw(raw_path)

    cleaned = load_and_clean(raw_path)
    splits = {"train": None, "val": None, "test": None}
    splits["train"], splits["val"], splits["test"] = stratified_split(cleaned)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_frame in splits.items():
        out_path = args.out_dir / f"{name}.csv"
        split_frame.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

    print("\nsplit sizes: " + "  ".join(f"{n}={len(s)}" for n, s in splits.items()))
    print_distribution_table(splits, "category", VALID_CATEGORIES)
    print_distribution_table(splits, "severity", VALID_SEVERITIES)


if __name__ == "__main__":
    main()
