from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_training_data.csv"
)

TACTICS = [
    "authority",
    "urgency",
    "threat",
    "fear",
    "impersonation",
    "credential_request",
    "payment_request",
    "reward_lure",
]


def main():

    print("=" * 70)
    print("SCAMSHIELD - TF-IDF MULTI-LABEL BASELINE")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"\nTotal messages: {len(df)}")

    # ---------------------------------------------------------
    # IMPORTANT:
    # Unknown labels (-1 / NaN) are NOT negative labels.
    #
    # For this first baseline we only use messages where every
    # tactic has a known human/pseudo label.
    # ---------------------------------------------------------

    complete_mask = df[TACTICS].notna().all(axis=1)

    complete_df = df[
        complete_mask
    ].copy()

    print(
        f"Messages with complete labels: "
        f"{len(complete_df)}"
    )

    if len(complete_df) < 20:

        raise ValueError(
            "Too few completely labelled messages "
            "for a meaningful baseline."
        )

    X = complete_df["message"].fillna("").astype(str)

    Y = complete_df[TACTICS].astype(int)

    # ---------------------------------------------------------
    # Display class distribution
    # ---------------------------------------------------------

    print("\nClass distribution:")

    for tactic in TACTICS:

        positives = int(
            Y[tactic].sum()
        )

        negatives = int(
            len(Y) - positives
        )

        print(
            f"{tactic:<25}"
            f"Positive: {positives:<4}"
            f"Negative: {negatives}"
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                )
            ),
        ),
    ])

    # ---------------------------------------------------------
    # Cross-validation
    #
    # We use KFold rather than StratifiedKFold because this is
    # a multi-label problem.
    # ---------------------------------------------------------

    n_splits = min(
        5,
        len(complete_df)
    )

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    print(
        f"\nRunning {n_splits}-fold cross-validation..."
    )

    scoring = {
        "micro_f1": "f1_micro",
        "macro_f1": "f1_macro",
        "micro_precision": "precision_micro",
        "micro_recall": "recall_micro",
    }

    scores = cross_validate(
        model,
        X,
        Y,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    for metric in scoring:

        values = scores[
            f"test_{metric}"
        ]

        print(
            f"{metric:<20}"
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )

    # ---------------------------------------------------------
    # Train final baseline on all complete data
    # ---------------------------------------------------------

    print(
        "\nTraining final baseline on all "
        "complete labelled messages..."
    )

    model.fit(
        X,
        Y
    )

    # ---------------------------------------------------------
    # Training-set diagnostic
    #
    # This is NOT a validation score.
    # It only confirms that the pipeline works.
    # ---------------------------------------------------------

    predictions = model.predict(X)

    print("\n" + "=" * 70)
    print("TRAINING-SET DIAGNOSTIC")
    print("=" * 70)

    print(
        classification_report(
            Y,
            predictions,
            target_names=TACTICS,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    import joblib

    MODEL_DIR = (
        ROOT
        / "models"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_PATH = (
        MODEL_DIR
        / "tfidf_multilabel_baseline.joblib"
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()