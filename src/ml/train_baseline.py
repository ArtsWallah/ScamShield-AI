
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "final_training_data.csv"
MODEL_PATH = ROOT / "models" / "tfidf_multilabel_baseline.joblib"

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


def make_classifier():
    return Pipeline([
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
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def main():
    print("=" * 70)
    print("SCAMSHIELD - PARTIALLY LABELLED MULTI-TACTIC TRAINING")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"\nTotal messages: {len(df)}")

    models = {}
    thresholds = {}
    evaluation_rows = []

    for tactic in TACTICS:
        print(f"\n{'-' * 60}\nTACTIC: {tactic}\n{'-' * 60}")

        known = df[df[tactic].notna()].copy()
        known[tactic] = pd.to_numeric(known[tactic], errors="raise").astype(int)

        if not known[tactic].isin([0, 1]).all():
            raise ValueError(f"Invalid labels found for {tactic}")

        X = known["message"].fillna("").astype(str)
        y = known[tactic]

        counts = y.value_counts().to_dict()
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())

        print(
            f"Known: {len(known)} | "
            f"Positive: {positives} | Negative: {negatives}"
        )

        if len(counts) < 2:
            print("SKIPPED: requires both positive and negative examples.")
            continue

        # Evaluate only rows whose source is fully human.
        # Mixed and pseudo-labelled rows are reserved for final fitting.
        if "label_source" in known.columns:
            eval_df = known[known["label_source"] == "human"].copy()
        else:
            eval_df = known.copy()

        eval_y = eval_df[tactic].astype(int)
        eval_counts = eval_y.value_counts()

        if len(eval_counts) == 2 and eval_counts.min() >= 2:
            folds = min(5, int(eval_counts.min()))
            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )

            eval_model = make_classifier()
            pred = cross_val_predict(
                eval_model,
                eval_df["message"].fillna("").astype(str),
                eval_y,
                cv=cv,
                method="predict",
            )

            print(f"Human-only {folds}-fold CV:")
            print(f"  F1:        {f1_score(eval_y, pred, zero_division=0):.3f}")
            print(
                f"  Precision: "
                f"{precision_score(eval_y, pred, zero_division=0):.3f}"
            )
            print(
                f"  Recall:    "
                f"{recall_score(eval_y, pred, zero_division=0):.3f}"
            )
        else:
            print(
                "CV skipped: too few human-labelled examples in one class "
                "for reliable stratified evaluation."
            )

        if positives < 5:
            print(
                "WARNING: fewer than 5 positive examples; "
                "predictions for this tactic will be highly uncertain."
            )

        model = make_classifier()
        model.fit(X, y)

        models[tactic] = model
        thresholds[tactic] = 0.5

    if not models:
        raise ValueError("No tactic had both positive and negative labels.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Store a dictionary so predict.py can load each tactic classifier.
    artifact = {
        "model_type": "per_tactic_binary_v1",
        "tactics": TACTICS,
        "models": models,
        "thresholds": thresholds,
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Trained classifiers:", len(models), "/", len(TACTICS))
    print("Saved model:", MODEL_PATH)
    print(
        "\nThese are experimental classifiers. "
        "Rare classes need more reviewed examples."
    )


if __name__ == "__main__":
    main()
