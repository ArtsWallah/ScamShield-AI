from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "human_annotations.csv"
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
    print("SCAMSHIELD - ANNOTATION QUALITY CHECK")
    print("=" * 70)

    if not INPUT_PATH.exists():

        print("\nERROR: Annotation file not found:")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal annotated messages: {len(df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "id",
        "message",
        *[f"label_{t}" for t in TACTICS],
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print("\nERROR: Missing columns:")

        for col in missing:
            print(" -", col)

        return

    print("\nRequired columns: OK")

    # --------------------------------------------------------
    # Convert labels safely
    # --------------------------------------------------------

    for tactic in TACTICS:

        column = f"label_{tactic}"

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TACTIC DISTRIBUTION")
    print("=" * 70)

    for tactic in TACTICS:

        column = f"label_{tactic}"

        positive = (
            df[column] == 1
        ).sum()

        negative = (
            df[column] == 0
        ).sum()

        missing = (
            df[column].isna()
        ).sum()

        print(
            f"\n{tactic:<25}"
            f" Positive: {positive:<5}"
            f" Negative: {negative:<5}"
            f" Missing: {missing}"
        )

    # --------------------------------------------------------
    # Messages with multiple tactics
    # --------------------------------------------------------

    label_columns = [
        f"label_{t}"
        for t in TACTICS
    ]

    df["num_tactics"] = (
        df[label_columns]
        .fillna(0)
        .eq(1)
        .sum(axis=1)
    )

    print("\n" + "=" * 70)
    print("MULTI-LABEL DISTRIBUTION")
    print("=" * 70)

    print(
        df["num_tactics"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Messages with zero labels
    # --------------------------------------------------------

    zero_labels = (
        df["num_tactics"] == 0
    ).sum()

    print(
        f"\nMessages with no selected tactic: "
        f"{zero_labels}"
    )

    # --------------------------------------------------------
    # Messages with multiple tactics
    # --------------------------------------------------------

    multi_label = (
        df["num_tactics"] > 1
    ).sum()

    print(
        f"Messages with multiple tactics: "
        f"{multi_label}"
    )

    # --------------------------------------------------------
    # Dataset readiness
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("READINESS CHECK")
    print("=" * 70)

    minimum_examples = 10

    ready = True

    for tactic in TACTICS:

        positive = (
            df[f"label_{tactic}"] == 1
        ).sum()

        if positive < minimum_examples:

            print(
                f"⚠ {tactic}: only "
                f"{positive} positive examples"
            )

            ready = False

        else:

            print(
                f"✓ {tactic}: "
                f"{positive} positive examples"
            )

    print("\n" + "=" * 70)

    if ready:

        print(
            "STATUS: Dataset is ready for baseline training."
        )

    else:

        print(
            "STATUS: More annotations are recommended "
            "before training."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()