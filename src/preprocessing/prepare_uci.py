from pathlib import Path
import pandas as pd


# Project root
ROOT = Path(__file__).resolve().parents[2]

# Input/output paths
INPUT_FILE = ROOT / "data" / "external" / "uci_sms" / "SMSSpamCollection"
OUTPUT_FILE = ROOT / "data" / "processed" / "uci_sms_clean.csv"


def main():
    print("Loading UCI SMS Spam Collection...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    print("\nOriginal shape:")
    print(df.shape)

    print("\nOriginal labels:")
    print(df["label"].value_counts())

    # Remove missing values
    df = df.dropna(subset=["message", "label"])

    # Remove exact duplicate messages
    before = len(df)
    df = df.drop_duplicates(subset=["message"])
    after = len(df)

    print(f"\nRemoved duplicates: {before - after}")

    # Convert labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Add dataset source
    df["source"] = "UCI_SMS"

    # Reset IDs
    df.insert(0, "id", range(1, len(df) + 1))

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nFinal shape:")
    print(df.shape)

    print("\nFinal labels:")
    print(df["label"].value_counts())

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()