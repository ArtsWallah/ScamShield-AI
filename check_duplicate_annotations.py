
import pandas as pd
from pathlib import Path

path = Path("data/processed/human_annotations.csv")
df = pd.read_csv(path)

label_columns = [
    column for column in df.columns
    if column.startswith("label_")
]

duplicates = df[
    df["message"].duplicated(keep=False)
].sort_values("message")

print("Rows in duplicate-message groups:", len(duplicates))
print("Duplicate message groups:", duplicates["message"].nunique())

if duplicates.empty:
    print("\nNo duplicate messages found.")
else:
    print("\nDuplicate messages:")
    print(
        duplicates[["id", "message"] + label_columns]
        .to_string(index=False, max_colwidth=70)
    )

    print("\nConflicting groups:")
    conflict_count = 0

    for message, group in duplicates.groupby("message"):
        conflicts = [
            column
            for column in label_columns
            if group[column].dropna().nunique() > 1
        ]

        if conflicts:
            conflict_count += 1
            print(
                f"IDs: {group['id'].tolist()} | "
                f"Conflicting labels: {conflicts}"
            )

    print("Total conflicting groups:", conflict_count)
