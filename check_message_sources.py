
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
processed = root / "data" / "processed"

missing_ids = [
    4210, 1623, 3558, 4730, 511, 583, 755, 4757, 4988,
    2270, 3332, 120, 121, 524, 585, 640, 785, 850
]

files = [
    "weak_class_annotation_queue.csv",
    "human_annotation_queue.csv",
    "fused_candidates.csv",
    "uci_sms_clean.csv",
    "training_dataset.csv",
    "annotation_candidates.csv",
    "annotation_candidates_v2.csv",
]

for filename in files:
    path = processed / filename

    if not path.exists():
        continue

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"\nCould not read {filename}: {exc}")
        continue

    print(f"\n{'=' * 65}")
    print(f"FILE: {filename}")
    print("Columns:", df.columns.tolist())

    id_col = next(
        (c for c in ["id", "ID", "message_id", "sms_id"] if c in df.columns),
        None,
    )

    if id_col is None:
        print("No recognized ID column; not matching rows automatically.")
        continue

    ids = pd.to_numeric(df[id_col], errors="coerce")
    matches = df[ids.isin(missing_ids)]

    if matches.empty:
        print("No matching IDs.")
    else:
        print(matches.to_string(index=False, max_colwidth=100))
