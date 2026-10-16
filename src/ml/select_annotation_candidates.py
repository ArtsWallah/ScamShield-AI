from pathlib import Path

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "tfidf_multilabel_baseline.joblib"
CANDIDATES_PATH = ROOT / "data" / "processed" / "fused_candidates.csv"
HUMAN_PATH = ROOT / "data" / "processed" / "human_annotations.csv"

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "active_learning_queue.csv"
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

# Classes where the current dataset is particularly weak.
TARGET_TACTICS = {
    "threat": 1,
    "fear": 1,
    "credential_request": 3,
    "impersonation": 6,
    "authority": 8,
    "payment_request": 9,
}


def main():

    print("=" * 70)
    print("SCAMSHIELD - ACTIVE LEARNING CANDIDATE SELECTOR")
    print("=" * 70)

    model = joblib.load(MODEL_PATH)

    candidates = pd.read_csv(
        CANDIDATES_PATH
    )

    human = pd.read_csv(
        HUMAN_PATH
    )

    print(
        f"\nFused candidates: {len(candidates)}"
    )

    print(
        f"Human annotations: {len(human)}"
    )

    # ---------------------------------------------------------
    # Remove already human-annotated IDs
    # ---------------------------------------------------------

    human_ids = set(
        pd.to_numeric(
            human["id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
    )

    candidates["id"] = pd.to_numeric(
        candidates["id"],
        errors="coerce"
    )

    candidates = candidates[
        ~candidates["id"].isin(human_ids)
    ].copy()

    candidates = candidates.dropna(
        subset=["message"]
    )

    print(
        f"Unannotated candidates: "
        f"{len(candidates)}"
    )

    # ---------------------------------------------------------
    # Model probabilities
    # ---------------------------------------------------------

    messages = (
        candidates["message"]
        .astype(str)
        .tolist()
    )

    probabilities = model.predict_proba(
        messages
    )

    # ---------------------------------------------------------
    # Build candidate records
    # ---------------------------------------------------------

    records = []

    for index, row in candidates.iterrows():

        message = str(
            row["message"]
        )

        probs = probabilities[
            list(candidates.index).index(index)
        ]

        # -----------------------------------------------------
        # Find uncertainty.
        #
        # Probability near 0.5 = model is uncertain.
        # -----------------------------------------------------

        uncertainty = (
            1.0
            -
            np.mean(
                np.abs(
                    probs - 0.5
                )
            ) * 2
        )

        # -----------------------------------------------------
        # Determine which weak tactics are most uncertain.
        # -----------------------------------------------------

        tactic_scores = {}

        for i, tactic in enumerate(TACTICS):

            if tactic not in TARGET_TACTICS:
                continue

            p = float(
                probs[i]
            )

            # Highest value around 0.5.
            tactic_uncertainty = (
                1.0
                -
                abs(
                    p - 0.5
                ) * 2
            )

            tactic_scores[
                tactic
            ] = tactic_uncertainty

        if not tactic_scores:
            continue

        best_tactic = max(
            tactic_scores,
            key=tactic_scores.get
        )

        best_tactic_uncertainty = (
            tactic_scores[
                best_tactic
            ]
        )

        # -----------------------------------------------------
        # Existing fusion information
        # -----------------------------------------------------

        fusion_score = float(
            row.get(
                "fusion_score",
                0
            )
            or 0
        )

        # -----------------------------------------------------
        # Active-learning priority
        #
        # High uncertainty + reasonable candidate quality.
        # -----------------------------------------------------

        priority = (
            0.70
            * best_tactic_uncertainty
            +
            0.30
            * min(
                fusion_score,
                1.0
            )
        )

        records.append({
            "id": int(row["id"]),
            "message": message,
            "target_tactic": best_tactic,
            "model_probability": float(
                probs[
                    TACTICS.index(
                        best_tactic
                    )
                ]
            ),
            "tactic_uncertainty":
                best_tactic_uncertainty,
            "overall_uncertainty":
                uncertainty,
            "fusion_score":
                fusion_score,
            "priority":
                priority,
        })

    queue = pd.DataFrame(
        records
    )

    if queue.empty:

        print(
            "\nNo active-learning candidates found."
        )
        return

    # ---------------------------------------------------------
    # Remove near-duplicates by message
    # ---------------------------------------------------------

    queue = queue.drop_duplicates(
        subset=["message"]
    )

    # ---------------------------------------------------------
    # Highest priority first
    # ---------------------------------------------------------

    queue = queue.sort_values(
        "priority",
        ascending=False
    )

    # ---------------------------------------------------------
    # Balance the queue across weak tactics
    #
    # Maximum 5 examples per tactic.
    # ---------------------------------------------------------

    selected = []

    for tactic in TARGET_TACTICS:

        tactic_rows = queue[
            queue["target_tactic"]
            == tactic
        ].head(5)

        selected.append(
            tactic_rows
        )

    queue = pd.concat(
        selected,
        ignore_index=True
    )

    queue = queue.sort_values(
        "priority",
        ascending=False
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    queue.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACTIVE LEARNING QUEUE")
    print("=" * 70)

    print(
        f"\nSelected candidates: "
        f"{len(queue)}"
    )

    print("\nCandidates by target tactic:")

    print(
        queue[
            "target_tactic"
        ].value_counts()
    )

    print("\nTop candidates:")

    print(
        queue[
            [
                "id",
                "target_tactic",
                "model_probability",
                "tactic_uncertainty",
                "priority",
                "message",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("ACTIVE LEARNING SELECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()