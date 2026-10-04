from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "human_annotation_queue.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "human_annotations.csv"
)


# ============================================================
# CONFIG
# ============================================================

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
    "credential_request": "Credential Request",
    "payment_request": "Payment Request",
    "reward_lure": "Reward Lure",
}


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ScamShield Annotation",
    page_icon="🛡️",
    layout="wide",
)


st.title("🛡️ ScamShield AI")
st.subheader("Human-in-the-Loop Manipulation Annotation")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not INPUT_PATH.exists():

        st.error(
            f"Annotation queue not found:\n{INPUT_PATH}"
        )

        st.stop()

    return pd.read_csv(INPUT_PATH)


df = load_data()


# ============================================================
# SESSION STATE
# ============================================================

if "index" not in st.session_state:

    st.session_state.index = 0


if "annotations" not in st.session_state:

    st.session_state.annotations = {}


# ============================================================
# CURRENT MESSAGE
# ============================================================

index = st.session_state.index

if index >= len(df):

    st.success(
        "🎉 Annotation complete!"
    )

    st.write(
        f"You annotated {len(df)} messages."
    )

    st.stop()


row = df.iloc[index]


# ============================================================
# PROGRESS
# ============================================================

progress = index / len(df)

st.progress(progress)

st.write(
    f"Message **{index + 1} / {len(df)}**"
)


# ============================================================
# MESSAGE
# ============================================================

st.markdown("### 📩 Message")

st.info(
    row["message"]
)


# ============================================================
# MODEL EVIDENCE
# ============================================================

st.markdown("### 🔍 Model Evidence")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Fusion Score",
        f"{row['fusion_score']:.3f}"
    )


with col2:

    st.metric(
        "Evidence",
        row["evidence_type"]
    )


with col3:

    st.metric(
        "Semantic Score",
        f"{row['semantic_score_normalized']:.3f}"
    )


if str(row["matched_keywords"]).strip():

    st.write(
        "**Keyword signals:**",
        row["matched_keywords"]
    )

else:

    st.write(
        "**Keyword signals:** None"
    )


# ============================================================
# ANNOTATION
# ============================================================

st.markdown("### 🧠 What manipulation tactics are present?")


selected = []


for tactic in TACTICS:

    key = f"{index}_{tactic}"

    checked = st.checkbox(
        TACTIC_LABELS[tactic],
        key=key
    )

    if checked:

        selected.append(tactic)


# ============================================================
# NOTES
# ============================================================

notes = st.text_area(
    "Annotation notes (optional)",
    placeholder=(
        "Explain why you selected the tactics..."
    )
)


annotator = st.text_input(
    "Annotator name / ID",
    value="student"
)


# ============================================================
# SAVE
# ============================================================

if st.button(
    "💾 Save Annotation & Next",
    type="primary"
):

    annotation = {
        "id": row["id"],
        "message": row["message"],
        "tactic_source": row["tactic"],
        "fusion_score": row["fusion_score"],
        "evidence_type": row["evidence_type"],
        "selected_tactics": "|".join(selected),
        "annotation_notes": notes,
        "annotator": annotator,
    }

    for tactic in TACTICS:

        annotation[
            f"label_{tactic}"
        ] = int(
            tactic in selected
        )

    annotation["annotation_status"] = "completed"

    st.session_state.annotations[index] = annotation

    # --------------------------------------------------------
    # Save all completed annotations
    # --------------------------------------------------------

    completed = list(
        st.session_state.annotations.values()
    )

    annotation_df = pd.DataFrame(
        completed
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Next message
    # --------------------------------------------------------

    st.session_state.index += 1

    st.rerun()


# ============================================================
# SKIP
# ============================================================

if st.button("⏭️ Skip"):

    st.session_state.index += 1

    st.rerun()