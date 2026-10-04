from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models"
    / "tfidf_multilabel_baseline.joblib"
)

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

    prediction = model.predict(
        [message]
    )[0]

    probabilities = model.predict_proba(
        [message]
    )[0]

    return prediction, probabilities


def main():

    print("=" * 70)
    print("SCAMSHIELD - SMS TACTIC PREDICTOR")
    print("=" * 70)

    print(
        "\nType an SMS message to analyse."
    )

    print(
        "Type 'exit' to stop.\n"
    )

    # ---------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    while True:

        message = input(
            "\nSMS > "
        ).strip()

        if message.lower() == "exit":
            break

        if not message:
            print(
                "Please enter a message."
            )
            continue

        prediction, probabilities = (
            predict_message(
                model,
                message
            )
        )

        print("\n" + "-" * 70)
        print("PREDICTION")
        print("-" * 70)

        detected = []

        for i, tactic in enumerate(TACTICS):

            score = float(
                probabilities[i]
            )

            label = int(
                prediction[i]
            )

            status = (
                "YES"
                if label == 1
                else "NO"
            )

            print(
                f"{tactic:<25}"
                f"{status:<6}"
                f"score={score:.3f}"
            )

            if label == 1:
                detected.append(
                    tactic
                )

        print("-" * 70)

        if detected:

            print(
                "Detected tactics:"
            )

            for tactic in detected:
                print(
                    f"  ✓ {tactic}"
                )

        else:

            print(
                "No manipulation tactic detected."
            )

        print("-" * 70)


if __name__ == "__main__":
    main()