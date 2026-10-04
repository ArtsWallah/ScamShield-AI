from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "uci_sms_clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "manipulation_annotations.csv"


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

    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    # Only use messages already labeled as spam/scam
    scam_df = df[df["label"] == 1].copy()

    print(f"Total messages: {len(df)}")
    print(f"Scam messages: {len(scam_df)}")

    # Create annotation columns
    for tactic in TACTICS:
        scam_df[tactic] = ""

    # Annotation status
    scam_df["annotation_status"] = "pending"

    # Keep useful columns
    scam_df = scam_df[
        [
            "id",
            "message",
            "label",
            *TACTICS,
            "annotation_status",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    scam_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\nAnnotation file created:")
    print(OUTPUT_PATH)

    print("\nColumns:")
    print(list(scam_df.columns))

    print("\nFirst 5 rows:")
    print(scam_df.head())


if __name__ == "__main__":
    main()