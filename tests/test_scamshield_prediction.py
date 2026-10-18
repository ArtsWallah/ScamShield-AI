
from pathlib import Path

import joblib
import numpy as np

from src.ml.predict import predict_message, TACTICS


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf_multilabel_baseline.joblib"


def load_artifact():
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_model_loads():
    artifact = load_artifact()
    assert artifact["model_type"] == "per_tactic_binary_v1"
    assert len(artifact["models"]) > 0


def test_prediction_returns_eight_outputs():
    artifact = load_artifact()
    prediction, scores = predict_message(
        artifact,
        "Your account needs verification. Please review your message."
    )

    assert len(prediction) == len(TACTICS) == 8
    assert len(scores) == 8
    assert set(np.asarray(prediction).tolist()).issubset({0, 1})
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_representative_messages_do_not_crash():
    artifact = load_artifact()

    messages = [
        "Congratulations! You have won a prize. Call to claim your reward.",
        "Your bank account will be suspended. Enter your password to restore access.",
        "Hey, are you coming to class tomorrow?",
    ]

    for message in messages:
        prediction, scores = predict_message(artifact, message)
        assert len(prediction) == 8
        assert len(scores) == 8


def test_empty_message_does_not_crash_model():
    artifact = load_artifact()
    prediction, scores = predict_message(artifact, "")

    assert len(prediction) == 8
    assert len(scores) == 8
