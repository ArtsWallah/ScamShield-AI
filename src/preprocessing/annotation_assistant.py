from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "manipulation_annotations.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "annotation_candidates_v2.csv"


# ------------------------------------------------------------
# Weighted manipulation signals
# ------------------------------------------------------------

TACTIC_SIGNALS = {

    "authority": {
        "bank": 3,
        "government": 3,
        "police": 3,
        "hmrc": 3,
        "official": 2,
        "department": 2,
        "tax office": 3,
        "revenue": 2,
        "customs": 2,
    },

    "urgency": {
        "immediately": 3,
        "urgent": 3,
        "now": 2,
        "asap": 3,
        "hurry": 3,
        "within": 2,
        "minutes": 2,
        "today": 1,
        "expires": 3,
        "deadline": 3,
    },

    "threat": {
        "blocked": 3,
        "suspended": 3,
        "deactivated": 3,
        "terminated": 3,
        "arrest": 4,
        "legal action": 4,
        "fine": 3,
        "penalty": 3,
        "prosecution": 4,
        "court": 3,
    },

    "fear": {
        "fraud": 2,
        "hacked": 3,
        "hacked account": 4,
        "unauthorized": 3,
        "security alert": 4,
        "suspicious activity": 4,
        "stolen": 3,
        "danger": 3,
        "risk": 2,
    },

    "impersonation": {
        "we are from": 4,
        "this is your bank": 5,
        "your bank": 3,
        "hmrc": 4,
        "police department": 5,
        "government department": 5,
        "official representative": 4,
        "customer service": 1,
    },

    "credential_request": {
        "otp": 5,
        "one time password": 5,
        "password": 5,
        "pin": 5,
        "passcode": 5,
        "verification code": 5,
        "security code": 4,
        "login details": 5,
        "username": 3,
        "credentials": 5,
        "cvv": 5,
    },

    "payment_request": {
        "transfer money": 5,
        "send money": 5,
        "make a payment": 4,
        "payment": 2,
        "transfer": 3,
        "fee": 2,
        "deposit": 3,
        "cash": 1,
        "charge": 2,
        "credit card": 3,
        "debit card": 3,
    },

    "reward_lure": {
        "winner": 4,
        "won": 4,
        "prize": 4,
        "reward": 3,
        "cash prize": 5,
        "free": 2,
        "bonus": 3,
        "gift": 3,
        "claim your prize": 5,
        "congratulations": 3,
        "jackpot": 5,
    },
}


def calculate_score(message, signals):

    message = message.lower()

    score = 0
    matched = []

    for phrase, weight in signals.items():

        if phrase in message:

            score += weight
            matched.append(phrase)

    return score, matched


def main():

    print("=" * 65)
    print("SCAMSHIELD - ANNOTATION ASSISTANT V2")
    print("Weighted + Auditable Candidate Generation")
    print("=" * 65)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded {len(df)} scam messages.")

    results = []

    for _, row in df.iterrows():

        message = str(row["message"])

        for tactic, signals in TACTIC_SIGNALS.items():

            score, matched = calculate_score(
                message,
                signals
            )

            if score > 0:

                results.append({
                    "id": row["id"],
                    "message": message,
                    "tactic": tactic,
                    "candidate_score": score,
                    "matched_keywords": " | ".join(matched),
                    "human_label": "",
                })

    candidates = pd.DataFrame(results)

    candidates = candidates.sort_values(
        ["tactic", "candidate_score"],
        ascending=[True, False]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(f"\nGenerated {len(candidates)} candidate annotations.")

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\nCandidates by tactic:")
    print(
        candidates["tactic"].value_counts()
    )

    print("\n" + "=" * 65)
    print("TOP CANDIDATES BY TACTIC")
    print("=" * 65)

    for tactic in TACTIC_SIGNALS:

        subset = candidates[
            candidates["tactic"] == tactic
        ].head(3)

        print(f"\n--- {tactic.upper()} ---")

        if subset.empty:
            print("No candidates found.")
            continue

        for _, row in subset.iterrows():

            print(
                f"\nID: {row['id']}"
                f"\nScore: {row['candidate_score']}"
                f"\nSignals: {row['matched_keywords']}"
                f"\nMessage: {row['message']}"
            )


if __name__ == "__main__":
    main()