from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pseudo_labeled_tactics.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_dataset.csv"
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
    print("SCAMSHIELD - TRAINING DATASET PREPARATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nInput rows: {len(df)}")

    # ---------------------------------------------------------
    # Keep only rows that contain useful labels
    # ---------------------------------------------------------

    label_columns = [
        f"label_{tactic}"
        for tactic in TACTICS
    ]

    df["has_label"] = df[
        label_columns
    ].notna().any(axis=1)

    df = df[
        df["has_label"]
    ].copy()

    print(
        f"Rows containing at least one label: "
        f"{len(df)}"
    )

    # ---------------------------------------------------------
    # Convert labels to numeric
    # ---------------------------------------------------------

    for column in label_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # Group by message ID
    #
    # Multiple tactic rows belonging to the same message
    # become ONE multi-label training example.
    # ---------------------------------------------------------

    training_rows = []

    for message_id, group in df.groupby(
        "id",
        sort=False
    ):

        message = (
            group["message"]
            .dropna()
            .astype(str)
            .iloc[0]
        )

        row = {
            "id": message_id,
            "message": message,
        }

        valid_label_count = 0

        for tactic in TACTICS:

            column = f"label_{tactic}"

            values = group[column].dropna()

            if len(values) == 0:

                row[tactic] = np.nan

            else:

                # If any trusted row says 1,
                # retain positive.
                if (values == 1).any():

                    row[tactic] = 1

                # Otherwise if we have a 0,
                # retain negative.
                elif (values == 0).any():

                    row[tactic] = 0

                else:

                    row[tactic] = np.nan

            if not pd.isna(row[tactic]):

                valid_label_count += 1

        row["num_labels"] = valid_label_count

        # -----------------------------------------------------
        # Track source
        # -----------------------------------------------------

        sources = (
            group["label_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if "human" in sources:

            row["label_source"] = "human"

        elif "pseudo" in sources:

            row["label_source"] = "pseudo"

        else:

            row["label_source"] = "unknown"

        training_rows.append(row)

    training = pd.DataFrame(
        training_rows
    )

    # ---------------------------------------------------------
    # Keep messages with at least one known tactic
    # ---------------------------------------------------------

    training = training[
        training["num_labels"] > 0
    ].copy()

    # ---------------------------------------------------------
    # Fill unknown labels with -1
    #
    # -1 means "unknown", NOT negative.
    #
    # This is important because missing labels must not
    # automatically become negative examples.
    # ---------------------------------------------------------

    for tactic in TACTICS:

        training[tactic] = (
            training[tactic]
            .fillna(-1)
            .astype(int)
        )

    # ---------------------------------------------------------
    # Remove exact duplicate messages
    # ---------------------------------------------------------

    training = training.drop_duplicates(
        subset=["message"],
        keep="first"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    training.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING DATASET")
    print("=" * 70)

    print(
        f"Unique training messages: "
        f"{len(training)}"
    )

    print("\nLabel source:")

    print(
        training["label_source"]
        .value_counts()
    )

    print("\nPositive labels:")

    for tactic in TACTICS:

        positives = (
            training[tactic] == 1
        ).sum()

        negatives = (
            training[tactic] == 0
        ).sum()

        unknown = (
            training[tactic] == -1
        ).sum()

        print(
            f"{tactic:<25}"
            f" Positive: {positives:<5}"
            f" Negative: {negatives:<5}"
            f" Unknown: {unknown}"
        )

    print("\nNumber of labels per message:")

    print(
        training["num_labels"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("TRAINING DATASET PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()