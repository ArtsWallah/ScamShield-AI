from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fused_candidates.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "human_annotation_queue.csv"
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

    print("=" * 65)
    print("SCAMSHIELD - HUMAN ANNOTATION QUEUE")
    print("=" * 65)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded candidates: {len(df)}")

    # --------------------------------------------------------
    # Prefer candidates where both systems agree
    # --------------------------------------------------------

    priority_order = {
        "dual": 0,
        "lexical_only": 1,
        "semantic_only": 2,
        "none": 3,
    }

    df["evidence_priority"] = (
        df["evidence_type"]
        .map(priority_order)
        .fillna(3)
    )

    # --------------------------------------------------------
    # Sort by evidence type and fusion score
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "evidence_priority",
            "fusion_score"
        ],
        ascending=[
            True,
            False
        ]
    )

    # --------------------------------------------------------
    # Remove duplicate messages
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["message"],
        keep="first"
    )

    print(
        f"Unique messages after "
        f"deduplication: {len(df)}"
    )

    # --------------------------------------------------------
    # Select annotation fields
    # --------------------------------------------------------

    columns = [
        "id",
        "message",
        "tactic",
        "fusion_score",
        "evidence_type",
        "lexical_score_normalized",
        "semantic_score_normalized",
        "matched_keywords",
    ]

    queue = df[columns].copy()

    # --------------------------------------------------------
    # Add human annotation columns
    # --------------------------------------------------------

    for tactic in TACTICS:

        queue[f"label_{tactic}"] = ""

    queue["annotation_notes"] = ""

    queue["annotator"] = ""

    queue["annotation_status"] = "pending"

    # --------------------------------------------------------
    # Add instructions
    # --------------------------------------------------------

    queue["annotation_instruction"] = (
        "Enter 1 if the tactic is clearly present, "
        "0 if clearly absent, leave blank if uncertain."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    queue.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(
        "\nAnnotation queue created:"
    )

    print(OUTPUT_PATH)

    print(
        f"\nTotal messages to annotate: "
        f"{len(queue)}"
    )

    print("\nAnnotation labels:")

    for tactic in TACTICS:
        print(
            f"  label_{tactic}"
        )

    print("\nRecommended values:")
    print("  1 = tactic present")
    print("  0 = tactic absent")
    print("  blank = uncertain")

    print("\nEvidence distribution:")

    print(
        queue["evidence_type"]
        .value_counts()
    )


if __name__ == "__main__":
    main()