
import pandas as pd
from pathlib import Path

processed = Path("data/processed")
annotations_path = processed / "human_annotations.csv"
decisions_path = processed / "scamshield_targeted_annotation_decisions.csv"
backup_path = processed / "human_annotations_backup.csv"

if not annotations_path.exists():
    raise FileNotFoundError(annotations_path)

if not decisions_path.exists():
    raise FileNotFoundError(
        f"{decisions_path} not found. Check the filename in data/processed."
    )

annotations = pd.read_csv(annotations_path)
decisions = pd.read_csv(decisions_path)

required = {"id", "target_tactic", "target_label"}
if not required.issubset(decisions.columns):
    raise ValueError(f"Decision CSV must contain: {required}")

if annotations["id"].duplicated().any():
    raise ValueError("Existing annotations contain duplicate IDs. Stopping.")

# Back up the original file before changing it.
if not backup_path.exists():
    annotations.to_csv(backup_path, index=False)

annotations = annotations.set_index("id", drop=False)

for _, decision in decisions.iterrows():
    row_id = decision["id"]
    tactic = str(decision["target_tactic"]).strip()
    label_column = f"label_{tactic}"

    if label_column not in annotations.columns:
        raise ValueError(
            f"Expected column '{label_column}' is missing from human_annotations.csv"
        )

    # Create a blank annotation row for genuinely new IDs.
    if row_id not in annotations.index:
        new_row = {column: pd.NA for column in annotations.columns}
        new_row["id"] = row_id

        if "annotation_status" in annotations.columns:
            new_row["annotation_status"] = "completed"

        annotations.loc[row_id] = new_row

    # Apply the targeted decision without overwriting other labels.
    annotations.loc[row_id, label_column] = int(decision["target_label"])

    if "annotation_status" in annotations.columns:
        annotations.loc[row_id, "annotation_status"] = "completed"

annotations = annotations.reset_index(drop=True)
annotations.to_csv(annotations_path, index=False)

print("Merge complete.")
print("Total rows:", len(annotations))
print("Unique IDs:", annotations["id"].nunique())
print("Backup:", backup_path)
