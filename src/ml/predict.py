
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "tfidf_multilabel_baseline.joblib"

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


def predict_message(model, message):
    # Compatibility with the new per-tactic model format.
    if isinstance(model, dict) and model.get("model_type") == "per_tactic_binary_v1":
        predictions = []
        probabilities = []

        for tactic in model["tactics"]:
            classifier = model["models"].get(tactic)

            if classifier is None:
                predictions.append(0)
                probabilities.append(0.0)
                continue

            probability = float(
                classifier.predict_proba([message])[0][
                    list(classifier.classes_).index(1)
                ]
            )

            threshold = model["thresholds"].get(tactic, 0.5)

            probabilities.append(probability)
            predictions.append(int(probability >= threshold))

        return np.array(predictions), np.array(probabilities)

    # Support the previous model format if an older model is loaded.
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    return prediction, probabilities


def main():
    print("=" * 70)
    print("SCAMSHIELD - SMS TACTIC PREDICTOR")
    print("=" * 70)
    print("\nType an SMS message to analyse.")
    print("Type 'exit' to stop.\n")

    model = joblib.load(MODEL_PATH)

    while True:
        message = input("\nSMS > ").strip()

        if message.lower() == "exit":
            break

        if not message:
            print("Please enter a message.")
            continue

        prediction, probabilities = predict_message(model, message)

        print("\n" + "-" * 70)
        print("EXPERIMENTAL PREDICTION")
        print("-" * 70)

        detected = []

        for i, tactic in enumerate(TACTICS):
            score = float(probabilities[i])
            label = int(prediction[i])

            print(
                f"{tactic:<25}"
                f"{'YES' if label else 'NO':<6}"
                f"score={score:.3f}"
            )

            if label:
                detected.append(tactic)

        print("-" * 70)

        if detected:
            print("Detected tactics:")
            for tactic in detected:
                print(f"  - {tactic}")
        else:
            print("No manipulation tactic detected.")

        print(
            "\nNote: scores are model estimates, not calibrated fraud "
            "probabilities. Rare tactics may be unreliable."
        )
        print("-" * 70)


if __name__ == "__main__":
    main()
