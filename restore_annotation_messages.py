
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
processed = root / "data" / "processed"

annotations_path = processed / "human_annotations.csv"
source_path = processed / "uci_sms_clean.csv"
backup_path = processed / "human_annotations_before_message_restore.csv"

annotations = pd.read_csv(annotations_path)
source = pd.read_csv(source_path)

if "id" not in annotations.columns or "message" not in annotations.columns:
    raise ValueError("Annotations must contain id and message columns.")

if not {"id", "message"}.issubset(source.columns):
    raise ValueError("Source dataset must contain id and message columns.")

annotations["id"] = pd.to_numeric(annotations["id"], errors="coerce")
source["id"] = pd.to_numeric(source["id"], errors="coerce")

source_messages = source.dropna(subset=["id"]).drop_duplicates("id").set_index("id")["message"]

missing_mask = annotations["message"].isna() | annotations["message"].astype(str).str.strip().eq("")
missing_ids = annotations.loc[missing_mask, "id"]

if missing_ids.isna().any():
    raise ValueError("Some rows with missing messages also have missing IDs.")

unmatched = [int(i) for i in missing_ids if i not in source_messages.index]

if unmatched:
    raise ValueError(f"Cannot find source messages for IDs: {unmatched}")

if not backup_path.exists():
    annotations.to_csv(backup_path, index=False)

for index in annotations.index[missing_mask]:
    message_id = annotations.at[index, "id"]
    annotations.at[index, "message"] = source_messages.loc[message_id]

annotations.to_csv(annotations_path, index=False)

print("Message restoration complete.")
print("Rows:", len(annotations))
print("Unique IDs:", annotations["id"].nunique())
print("Missing messages:", annotations["message"].isna().sum())
print("Backup:", backup_path)
