from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Paths and schema
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
NORMAL_QUEUE_PATH = DATA_DIR / "human_annotation_queue.csv"
WEAK_QUEUE_PATH = DATA_DIR / "weak_class_annotation_queue.csv"
OUTPUT_PATH = DATA_DIR / "human_annotations.csv"
HARD_NEGATIVE_QUEUE_PATH = DATA_DIR / "hard_negative_annotation_queue.csv"

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
TACTIC_LABELS = {
    "authority": "Authority",
    "urgency": "Urgency",
    "threat": "Threat",
    "fear": "Fear",
    "impersonation": "Impersonation",
    "credential_request": "Credential request",
    "payment_request": "Payment request",
    "reward_lure": "Reward lure",
}
LABEL_COLS = [f"label_{t}" for t in TACTICS]
UNKNOWN = "Unknown / not enough evidence"
YES = "Yes — clearly present"
NO = "No — clearly absent"
SKIP = "Skip for now"


st.set_page_config(page_title="ScamShield AI — Annotation", page_icon="🛡️", layout="wide")
st.title("🛡️ ScamShield AI")
st.caption("Human-in-the-loop annotation workspace · Unknown is not the same as No")

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"id": str}, keep_default_na=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    # Avoid IDs changing from "123" to "123.0" after CSV round trips.
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def label_value(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        number = int(float(value))
        return number if number in (0, 1) else None
    except (TypeError, ValueError):
        return None


def get_existing() -> pd.DataFrame:
    existing = read_csv_safe(OUTPUT_PATH)
    if existing.empty:
        return pd.DataFrame(columns=["id", "message", *LABEL_COLS])
    if "id" not in existing.columns:
        existing["id"] = ""
    existing["id"] = existing["id"].map(clean_id)
    # If a historical file contains duplicate IDs, keep the latest saved row.
    existing = existing[existing["id"] != ""].drop_duplicates("id", keep="last")
    for col in LABEL_COLS:
        if col not in existing.columns:
            existing[col] = None
        existing[col] = existing[col].map(label_value)
    return existing


def atomic_save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_PATH.with_suffix(".tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(OUTPUT_PATH)


def upsert_annotation(annotation: dict, *, overwrite_labels: set[str]) -> None:
    """Merge one decision into the existing file without erasing unrelated labels."""
    existing = get_existing()
    annotation_id = clean_id(annotation.get("id"))
    if not annotation_id:
        raise ValueError("This queue item has no ID; cannot save it safely.")

    if existing.empty:
        base = {column: None for column in ["id", "message", *LABEL_COLS]}
        existing = pd.DataFrame([base])

    match = existing.index[existing["id"].map(clean_id) == annotation_id].tolist()
    if match:
        idx = match[-1]
        # Update ordinary metadata only when a non-empty value is supplied.
        for key, value in annotation.items():
            if key in LABEL_COLS:
                if key in overwrite_labels:
                    existing.at[idx, key] = label_value(value)
            elif value is not None and not (isinstance(value, float) and math.isnan(value)):
                if str(value).strip() != "":
                    if key not in existing.columns:
                        existing[key] = None
                    existing.at[idx, key] = value
    else:
        row = {column: None for column in existing.columns}
        for key, value in annotation.items():
            if key not in existing.columns:
                existing[key] = None
                row[key] = None
            row[key] = value
        for col in LABEL_COLS:
            if col in row:
                row[col] = label_value(row[col])
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)

    # Keep a single row per ID and normalize IDs.
    existing["id"] = existing["id"].map(clean_id)
    existing = existing[existing["id"] != ""].drop_duplicates("id", keep="last")
    atomic_save(existing)


def row_for_id(existing: pd.DataFrame, item_id: str):
    if existing.empty or "id" not in existing.columns:
        return None
    matches = existing[existing["id"].map(clean_id) == clean_id(item_id)]
    return None if matches.empty else matches.iloc[-1]


def is_known(value) -> bool:
    return label_value(value) in (0, 1)


def metric_value(row, col):
    if col not in row.index:
        return None
    value = row[col]
    if pd.isna(value) or str(value).strip() == "":
        return None
    return value


def pretty_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def item_completed(item, existing, mode):
    saved = row_for_id(existing, clean_id(item.get("id", "")))
    if saved is None:
        return False
    if mode == "Target rare classes":
        target = str(item.get("target_tactic", "")).strip()
        return target in TACTICS and is_known(saved.get(f"label_{target}"))
    if mode == "Hard-negative review":
        # Review is complete when the annotator has explicitly decided all tactics.
        return all(is_known(saved.get(col)) for col in LABEL_COLS)
    return all(is_known(saved.get(col)) for col in LABEL_COLS)


# ---------------------------------------------------------------------
# Mode / data
# ---------------------------------------------------------------------
mode = st.sidebar.radio(
    "Workflow",
    ["Normal multi-label", "Target rare classes", "Hard-negative review"],
    help="Normal: annotate all tactics. Rare classes: decide only the target tactic. Hard negatives: review likely non-scam / low-evidence examples.",
)
annotator = st.sidebar.text_input("Annotator ID", value="student").strip() or "student"
show_completed = st.sidebar.checkbox("Include already completed items", value=False)


if mode == "Target rare classes":
    queue_path = WEAK_QUEUE_PATH
elif mode == "Hard-negative review":
    queue_path = HARD_NEGATIVE_QUEUE_PATH
else:
    queue_path = NORMAL_QUEUE_PATH

if not queue_path.exists():
    st.error(f"Queue file not found: `{queue_path}`")
    st.info("Generate the queue first, or check the project data/processed directory.")
    st.stop()

queue = read_csv_safe(queue_path)
if queue.empty:
    st.warning("The selected queue is empty.")
    st.stop()

if "id" not in queue.columns or "message" not in queue.columns:
    st.error(f"Queue must contain `id` and `message` columns. Found: {list(queue.columns)}")
    st.stop()

queue["id"] = queue["id"].map(clean_id)
queue = queue[queue["id"] != ""].drop_duplicates("id", keep="first").reset_index(drop=True)
existing = get_existing()

# Hard-negative review orders low-score candidates first; it does not assume they are negatives.
if mode == "Hard-negative review":
    score_col = next((c for c in ["fusion_score", "keyword_score", "semantic_score_normalized"] if c in queue.columns), None)
    if score_col:
        queue[score_col] = pd.to_numeric(queue[score_col], errors="coerce")
        queue = queue.sort_values(score_col, ascending=True, na_position="first").reset_index(drop=True)

eligible = queue.copy()
if not show_completed:
    eligible = eligible[
        ~eligible.apply(lambda item: item_completed(item, existing, mode), axis=1)
    ].reset_index(drop=True)

# Sidebar progress uses all unique IDs in this mode's queue, not the current filtered queue.
completed_count = sum(item_completed(item, existing, mode) for _, item in queue.iterrows())
total_count = len(queue)
st.sidebar.markdown("### Progress")
st.sidebar.progress(completed_count / total_count if total_count else 0.0)
st.sidebar.write(f"**{completed_count} / {total_count}** completed")
st.sidebar.caption(f"{total_count - completed_count} remaining in this queue")

if eligible.empty:
    st.success("Everything in this queue is complete.")
    st.write(f"Annotations are stored at `{OUTPUT_PATH}`.")
    st.stop()

# Navigation index is per workflow and clamped when queue changes.
state_key = "annotation_position_" + mode.replace(" ", "_")
if state_key not in st.session_state:
    st.session_state[state_key] = 0
position = int(st.session_state[state_key])
position = max(0, min(position, len(eligible) - 1))
st.session_state[state_key] = position
item = eligible.iloc[position]
item_id = clean_id(item["id"])
saved_row = row_for_id(existing, item_id)

st.progress((position + 1) / len(eligible))
st.caption(f"Current item {position + 1} of {len(eligible)} remaining items · ID {item_id}")

left, right = st.columns([3, 1])
with left:
    st.subheader("Message")
    st.info(str(item["message"]))
with right:
    st.subheader("Candidate evidence")
    shown = False
    for col, label in [
        ("target_tactic", "Target tactic"),
        ("tactic", "Queue tactic"),
        ("fusion_score", "Fusion score"),
        ("evidence_type", "Evidence type"),
        ("semantic_score_normalized", "Semantic score"),
        ("keyword_score", "Keyword score"),
        ("matched_keywords", "Matched keywords"),
        ("source", "Source"),
    ]:
        value = metric_value(item, col)
        if value is not None:
            st.write(f"**{label}:** {pretty_value(value)}")
            shown = True
    if not shown:
        st.caption("No extra evidence fields are available for this candidate.")

if mode == "Target rare classes":
    target = str(item.get("target_tactic", "")).strip()
    if target not in TACTICS:
        st.error(f"Unknown target tactic `{target}`. Expected one of: {', '.join(TACTICS)}")
        st.stop()
    st.markdown(f"### Target tactic: **{TACTIC_LABELS[target]}**")
    st.caption("Decide only whether this tactic is clearly present. Skip or unknown never becomes a negative.")
    old_value = None if saved_row is None else label_value(saved_row.get(f"label_{target}"))
    default = UNKNOWN if old_value is None else (YES if old_value == 1 else NO)
    decision = st.radio(
        "Decision",
        [UNKNOWN, YES, NO, SKIP],
        index=[UNKNOWN, YES, NO, SKIP].index(default),
        key=f"target_decision_{mode}_{item_id}",
    )
    notes = st.text_area("Notes (optional)", value="" if saved_row is None or pd.isna(saved_row.get("annotation_notes")) else str(saved_row.get("annotation_notes", "")), key=f"notes_{mode}_{item_id}")
    if st.button("Save target decision", type="primary", use_container_width=True):
        value = 1 if decision == YES else 0 if decision == NO else None
        status = "completed" if decision in (YES, NO) else "skipped" if decision == SKIP else "in_progress"
        annotation = {
            "id": item_id, "message": str(item["message"]), "tactic_source": target,
            "annotation_notes": notes, "annotator": annotator, "annotation_status": status,
            "annotation_mode": mode, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            f"label_{target}": value,
        }
        try:
            upsert_annotation(annotation, overwrite_labels={f"label_{target}"})
            st.success("Saved. Other tactic labels and existing annotations were preserved.")
            st.session_state[state_key] = min(position + 1, len(eligible) - 1)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save annotation: {exc}")

else:
    st.markdown("### Label each tactic")
    st.caption("Choose Unknown when evidence is insufficient. Only an explicit No is saved as 0.")
    choices = [UNKNOWN, YES, NO]
    selected_values = {}
    columns = st.columns(2)
    for i, tactic in enumerate(TACTICS):
        old_value = None if saved_row is None else label_value(saved_row.get(f"label_{tactic}"))
        default = UNKNOWN if old_value is None else (YES if old_value == 1 else NO)
        with columns[i % 2]:
            selected_values[tactic] = st.selectbox(
                TACTIC_LABELS[tactic], choices,
                index=choices.index(default),
                key=f"label_{mode}_{item_id}_{tactic}",
            )
    notes_default = ""
    if saved_row is not None and "annotation_notes" in saved_row.index and pd.notna(saved_row.get("annotation_notes")):
        notes_default = str(saved_row.get("annotation_notes"))
    notes = st.text_area("Notes (optional)", value=notes_default, key=f"notes_{mode}_{item_id}")
    if st.button("Save labels", type="primary", use_container_width=True):
        labels = {
            f"label_{tactic}": 1 if selected_values[tactic] == YES else 0 if selected_values[tactic] == NO else None
            for tactic in TACTICS
        }
        # Hard-negative review is complete only when every tactic has an explicit Yes/No.
        complete = all(value is not None for value in labels.values())
        annotation = {
            "id": item_id, "message": str(item["message"]),
            "tactic_source": str(item.get("tactic", "")),
            "selected_tactics": "|".join(t for t in TACTICS if labels[f"label_{t}"] == 1),
            "annotation_notes": notes, "annotator": annotator,
            "annotation_status": "completed" if complete else "in_progress",
            "annotation_mode": mode, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **labels,
        }
        try:
            upsert_annotation(annotation, overwrite_labels=set(LABEL_COLS))
            st.success("Saved. Unknown labels remain blank; existing rows were merged safely.")
            st.session_state[state_key] = min(position + 1, len(eligible) - 1)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save annotation: {exc}")

# Navigation controls
st.divider()
nav_left, nav_mid, nav_right = st.columns(3)
with nav_left:
    if st.button("← Previous", disabled=position == 0, use_container_width=True):
        st.session_state[state_key] = max(0, position - 1)
        st.rerun()
with nav_mid:
    if st.button("Skip without saving", use_container_width=True):
        st.session_state[state_key] = min(position + 1, len(eligible) - 1)
        st.rerun()
with nav_right:
    if st.button("Next →", disabled=position >= len(eligible) - 1, use_container_width=True):
        st.session_state[state_key] = min(position + 1, len(eligible) - 1)
        st.rerun()

with st.expander("Annotation guidance"):
    st.markdown(
        """
- **Yes:** the tactic is clearly expressed in the message.
- **No:** the message provides enough evidence to confidently say the tactic is absent.
- **Unknown:** there is not enough evidence to decide. Unknown is stored as a blank value, not `0`.
- **Skip without saving:** move past the item without changing its saved labels.
- **Target rare classes:** updates only the selected target tactic and preserves every other tactic label.
- **Hard-negative review:** presents lower-scoring candidates first; low score is not proof that a message is benign.
"""
    )

st.caption(f"Output file: `{OUTPUT_PATH}`")
