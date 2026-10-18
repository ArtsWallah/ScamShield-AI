from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

HUMAN_PATH = ROOT / "data" / "processed" / "human_annotations.csv"
V4_PATH = ROOT / "data" / "processed" / "pseudo_labeled_tactics.csv"
V5_PATH = ROOT / "data" / "processed" / "expanded_training_candidates.csv"

OUTPUT_PATH = ROOT / "data" / "processed" / "final_training_data.csv"
APPROVED_CORRECTIONS_PATH = (
    ROOT / "data" / "processed" / "approved_label_corrections.csv"
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
    print("SCAMSHIELD - FINAL TRAINING DATA BUILDER")
    print("=" * 70)

    human = pd.read_csv(HUMAN_PATH)
    v4 = pd.read_csv(V4_PATH)
    v5 = pd.read_csv(V5_PATH)

    print(f"\nHuman annotations: {len(human)}")
    print(f"V4 pseudo-label rows: {len(v4)}")
    print(f"V5 expansion rows: {len(v5)}")

    # ---------------------------------------------------------
    # Normalize IDs
    # ---------------------------------------------------------

    for df in [human, v4, v5]:

        df["id"] = pd.to_numeric(
            df["id"],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # Start with HUMAN annotations.
    #
    # Human labels always have highest priority.
    # ---------------------------------------------------------

    records = {}

    for _, row in human.iterrows():

        if pd.isna(row["id"]):
            continue

        message_id = int(row["id"])

        records[message_id] = {
            "id": message_id,
            "message": str(row["message"]),
            "label_source": "human",
        }

        for tactic in TACTICS:

            column = f"label_{tactic}"

            value = pd.to_numeric(
                row.get(column),
                errors="coerce"
            )

            if pd.isna(value):
                records[message_id][tactic] = np.nan
            else:
                records[message_id][tactic] = int(value)

    # ---------------------------------------------------------
    # Add V4 pseudo-labels.
    #
    # Only add labels that don't conflict with human labels.
    # ---------------------------------------------------------

    v4_added = 0

    for _, row in v4.iterrows():

        if row.get("label_source") != "pseudo":
            continue

        if not bool(
            row.get("training_eligible", False)
        ):
            continue

        if pd.isna(row["id"]):
            continue

        message_id = int(row["id"])

        tactic = str(
            row.get("tactic", "")
        ).strip()

        if tactic not in TACTICS:
            continue

        if message_id not in records:

            records[message_id] = {
                "id": message_id,
                "message": str(row["message"]),
                "label_source": "pseudo",
            }

            for t in TACTICS:
                records[message_id][t] = np.nan

        # Human label always wins.
        current = records[message_id].get(
            tactic,
            np.nan
        )

        if pd.isna(current):

            records[message_id][tactic] = 1

            if records[message_id][
                "label_source"
            ] != "human":

                records[message_id][
                    "label_source"
                ] = "human+pseudo"

            v4_added += 1

    # ---------------------------------------------------------
    # Add V5 candidates.
    # ---------------------------------------------------------

    v5_added = 0

    for _, row in v5.iterrows():

        if pd.isna(row["id"]):
            continue

        message_id = int(row["id"])

        tactic = str(
            row.get("tactic", "")
        ).strip()

        if tactic not in TACTICS:
            continue

        confidence = float(
            row.get(
                "pseudo_confidence",
                0
            )
            or 0
        )

        if confidence < 0.60:
            continue

        if message_id not in records:

            records[message_id] = {
                "id": message_id,
                "message": str(row["message"]),
                "label_source": "v5_pseudo",
            }

            for t in TACTICS:
                records[message_id][t] = np.nan

        current = records[message_id].get(
            tactic,
            np.nan
        )

        # Never overwrite a human label.
        if pd.isna(current):

            records[message_id][tactic] = 1

            source = records[
                message_id
            ]["label_source"]

            if source == "human":

                records[
                    message_id
                ]["label_source"] = "human+v5"

            elif source == "human+pseudo":

                records[
                    message_id
                ]["label_source"] = "human+pseudo+v5"

            else:

                records[
                    message_id
                ]["label_source"] = "v5_pseudo"

            v5_added += 1

    # ---------------------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------------------

    final_df = pd.DataFrame(
        list(records.values())
    )
    # ---------------------------------------------------------
    # Apply explicitly approved label corrections
    # ---------------------------------------------------------

    if not APPROVED_CORRECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Approved correction audit not found: "
            f"{APPROVED_CORRECTIONS_PATH}"
        )

    corrections = pd.read_csv(APPROVED_CORRECTIONS_PATH)

    required_audit_columns = {
        "id",
        "tactic",
        "current_label",
        "approved_label",
        "approval_status",
        "approval_basis",
    }

    missing_columns = required_audit_columns - set(corrections.columns)
    if missing_columns:
        raise ValueError(
            f"Correction audit is missing columns: {sorted(missing_columns)}"
        )

    if len(corrections) != 15:
        raise ValueError(
            f"Expected 15 approved correction records; found {len(corrections)}"
        )

    if not corrections["approval_status"].eq("APPROVED_BY_USER").all():
        raise ValueError("Every correction must be explicitly approved.")

    if corrections.duplicated(["id", "tactic"]).any():
        raise ValueError("Duplicate (id, tactic) entries in correction audit.")

    if not corrections["tactic"].isin(TACTICS).all():
        raise ValueError("Correction audit contains an unknown tactic.")

    corrections["id"] = pd.to_numeric(
        corrections["id"], errors="raise"
    ).astype(int)

    for _, correction in corrections.iterrows():
        message_id = int(correction["id"])
        tactic = str(correction["tactic"])
        expected = pd.to_numeric(
            pd.Series([correction["current_label"]]), errors="coerce"
        ).iloc[0]
        approved = pd.to_numeric(
            pd.Series([correction["approved_label"]]), errors="coerce"
        ).iloc[0]

        if pd.isna(expected) or expected not in (0, 1):
            raise ValueError(
                f"Invalid recorded original label for ID {message_id}, {tactic}"
            )

        if pd.isna(approved) or approved not in (0, 1):
            raise ValueError(
                f"Invalid approved label for ID {message_id}, {tactic}"
            )

        matches = final_df.index[final_df["id"].eq(message_id)].tolist()
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one merged record for ID {message_id}; "
                f"found {len(matches)}"
            )

        idx = matches[0]
        actual = pd.to_numeric(
            pd.Series([final_df.at[idx, tactic]]), errors="coerce"
        ).iloc[0]

        if pd.isna(actual) or actual != expected:
            raise ValueError(
                f"Stale/conflicting audit for ID {message_id}, {tactic}: "
                f"expected original {expected}, found {actual}"
            )

        final_df.at[idx, tactic] = int(approved)

        print(
            f"Approved correction applied: ID {message_id}, "
            f"{tactic}: {int(expected)} -> {int(approved)}"
        )

    # ---------------------------------------------------------
    # Ensure consistent columns
    # ---------------------------------------------------------

    columns = [
        "id",
        "message",
        *TACTICS,
        "label_source",
    ]

    final_df = final_df[
        columns
    ]

    # ---------------------------------------------------------
    # Remove duplicate messages
    # ---------------------------------------------------------

    final_df = final_df.drop_duplicates(
        subset=["message"],
        keep="first"
    )

    # ---------------------------------------------------------
    # Numeric labels
    # ---------------------------------------------------------

    for tactic in TACTICS:

        final_df[tactic] = pd.to_numeric(
            final_df[tactic],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # Number of known labels
    # ---------------------------------------------------------

    final_df["known_labels"] = (
        final_df[TACTICS]
        .notna()
        .sum(axis=1)
    )

    # ---------------------------------------------------------
    # Number of positive labels
    # ---------------------------------------------------------

    final_df["positive_labels"] = (
        final_df[TACTICS]
        .eq(1)
        .sum(axis=1)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MERGE RESULTS")
    print("=" * 70)

    print(
        f"\nV4 labels added: {v4_added}"
    )

    print(
        f"V5 labels added: {v5_added}"
    )

    print(
        f"Final unique messages: "
        f"{len(final_df)}"
    )

    print("\nLabel sources:")

    print(
        final_df[
            "label_source"
        ].value_counts()
    )

    print("\nTACTIC DISTRIBUTION")
    print("-" * 70)

    for tactic in TACTICS:

        positive = (
            final_df[tactic] == 1
        ).sum()

        negative = (
            final_df[tactic] == 0
        ).sum()

        unknown = (
            final_df[tactic].isna()
        ).sum()

        print(
            f"{tactic:<25}"
            f" Positive: {positive:<5}"
            f" Negative: {negative:<5}"
            f" Unknown: {unknown}"
        )

    print("\nLABELS PER MESSAGE")
    print("-" * 70)

    print(
        final_df[
            "positive_labels"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("FINAL TRAINING DATA BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()