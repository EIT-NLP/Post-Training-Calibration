#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read figure1_first_token_entropy_points.csv and compute AUROC."
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to figure1_first_token_entropy_points.csv",
    )
    parser.add_argument(
        "--score_col",
        default="first_token_entropy",
        help="Score column name. Default: first_token_entropy",
    )
    parser.add_argument(
        "--label_col",
        default="label_id",
        help="Label column name. Default: label_id",
    )
    parser.add_argument(
        "--save_clean_csv",
        default=None,
        help="Optional path to save the cleaned dataframe used for AUROC.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {args.score_col, args.label_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    work_df = df[[args.label_col, args.score_col]].copy()
    work_df = work_df.dropna()

    # In your CSV from the entropy script:
    # label_id == 0 means correct
    # label_id == 1 means wrong
    y_wrong = work_df[args.label_col].astype(int).to_numpy()
    entropy = work_df[args.score_col].astype(float).to_numpy()

    unique_labels = sorted(set(y_wrong.tolist()))
    if len(unique_labels) != 2:
        raise ValueError(
            f"AUROC needs exactly 2 classes, but got labels: {unique_labels}"
        )
    if unique_labels != [0, 1]:
        raise ValueError(
            f"Expected binary labels {{0,1}}, but got: {unique_labels}"
        )

    # 1) Use entropy to predict WRONG (positive class = wrong = 1)
    auroc_wrong_by_entropy = roc_auc_score(y_wrong, entropy)

    # 2) Use -entropy to predict CORRECT (positive class = correct = 1)
    y_correct = 1 - y_wrong
    auroc_correct_by_neg_entropy = roc_auc_score(y_correct, -entropy)

    # 3) Equivalent view: entropy for CORRECT directly
    #    This number is usually 1 - AUROC if there are no ties.
    auroc_correct_by_entropy = roc_auc_score(y_correct, entropy)

    print(f"Loaded rows after dropna: {len(work_df)}")
    print(f"Score column: {args.score_col}")
    print(f"Label column: {args.label_col} (0=correct, 1=wrong)")
    print()
    print(f"AUROC (entropy -> predict WRONG):   {auroc_wrong_by_entropy:.6f}")
    print(f"AUROC (-entropy -> predict CORRECT): {auroc_correct_by_neg_entropy:.6f}")
    print(f"AUROC (entropy -> predict CORRECT):  {auroc_correct_by_entropy:.6f}")

    if args.save_clean_csv is not None:
        out_path = Path(args.save_clean_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df = work_df.copy()
        clean_df["y_wrong"] = y_wrong
        clean_df["y_correct"] = y_correct
        clean_df["entropy"] = entropy
        clean_df.to_csv(out_path, index=False)
        print(f"Saved cleaned CSV: {out_path}")


if __name__ == "__main__":
    main()
