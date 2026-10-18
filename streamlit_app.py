
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tfidf_multilabel_baseline.joblib"
DATA_PATH = ROOT / "data" / "processed" / "final_training_data.csv"

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

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ScamShield AI")
st.subheader("SMS Social Engineering Analysis")
st.write(
    "Analyze an SMS for potential manipulation tactics using an "
    "experimental machine-learning model."
)

st.warning(
    "Research prototype only. Scores are not calibrated probabilities. "
    "A prediction cannot prove that a message is fraudulent or safe. "
    "Do not rely on this tool alone for financial or security decisions."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_positive_counts():
    if not DATA_PATH.exists():
        return {}

    df = pd.read_csv(DATA_PATH)
    return {
        tactic: int((pd.to_numeric(df[tactic], errors="coerce") == 1).sum())
        for tactic in TACTICS
        if tactic in df.columns
    }


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    artifact = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

if not (
    isinstance(artifact, dict)
    and artifact.get("model_type") == "per_tactic_binary_v1"
):
    st.error("Unsupported model format. Please retrain using train_baseline.py.")
    st.stop()

positive_counts = load_positive_counts()

st.markdown("### Analyze a message")

message = st.text_area(
    "SMS message",
    placeholder=(
        "Example: Your account will be suspended. "
        "Click the link and enter your password."
    ),
    height=150,
)

analyze = st.button(
    "Analyze SMS",
    type="primary",
    use_container_width=True,
)

if analyze:
    if not message.strip():
        st.error("Please enter an SMS message first.")
        st.stop()

    results = []

    for tactic in TACTICS:
        classifier = artifact.get("models", {}).get(tactic)

        if classifier is None:
            results.append({
                "Tactic": TACTIC_LABELS[tactic],
                "Status": "Unavailable",
                "Score": None,
                "Positive examples": positive_counts.get(tactic, 0),
            })
            continue

        try:
            probabilities = classifier.predict_proba([message])[0]
            classes = list(classifier.classes_)

            if 1 not in classes:
                score = 0.0
            else:
                score = float(probabilities[classes.index(1)])

            threshold = artifact.get("thresholds", {}).get(tactic, 0.5)
            predicted = score >= threshold

            results.append({
                "Tactic": TACTIC_LABELS[tactic],
                "Status": "Potential signal" if predicted else "Not flagged",
                "Score": score,
                "Positive examples": positive_counts.get(tactic, 0),
            })

        except Exception:
            results.append({
                "Tactic": TACTIC_LABELS[tactic],
                "Status": "Prediction error",
                "Score": None,
                "Positive examples": positive_counts.get(tactic, 0),
            })

    flagged = [
        result for result in results
        if result["Status"] == "Potential signal"
    ]

    st.markdown("---")
    st.markdown("### Analysis results")

    if flagged:
        st.warning(
            f"The model flagged {len(flagged)} potential tactic(s). "
            "Review the message carefully; this is not a fraud verdict."
        )
    else:
        st.info(
            "No tactics were flagged by the available classifiers. "
            "This does not mean the message is safe."
        )

    for result in results:
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.markdown(f"**{result['Tactic']}**")

        with col2:
            status = result["Status"]

            if status == "Potential signal":
                st.error(status)
            elif status == "Not flagged":
                st.success(status)
            else:
                st.warning(status)

        with col3:
            score = result["Score"]

            if score is None:
                st.write("Score unavailable")
            else:
                st.write(f"Model score: {score:.3f}")
                st.progress(max(0.0, min(1.0, score)))

        if result["Positive examples"] < 5:
            st.caption(
                f"Limited training evidence: only "
                f"{result['Positive examples']} positive example(s) "
                "in the current training dataset."
            )

    st.caption(
        "Scores are model outputs, not calibrated fraud probabilities. "
        "Training-data counts are not a substitute for independent evaluation."
    )

st.markdown("---")
st.caption(
    "ScamShield AI | Experimental multi-tactic SMS analysis | "
    "Use independent verification for suspicious messages."
)
