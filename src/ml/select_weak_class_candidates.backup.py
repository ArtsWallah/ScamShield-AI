from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_PATH = ROOT / "data" / "processed" / "fused_candidates.csv"
HUMAN_PATH = ROOT / "data" / "processed" / "human_annotations.csv"

OUTPUT_PATH = ROOT / "data" / "processed" / "weak_class_annotation_queue.csv"


TARGETS = {
    "threat": 10,
    "fear": 10,
    "credential_request": 10,
}


# Keywords are only used to find plausible candidates.
# They are NOT treated as labels.
KEYWORDS = {
    "threat": [
        "close",
        "closed",
        "blocked",
        "block",
        "cancel",
        "cancelled",
        "suspend",
        "suspended",
        "expire",
        "expired",
        "warning",
        "final",
        "last chance",
        "action required",
        "failure",
    ],

    "fear": [
        "danger",
        "warning",
        "risk",
        "security",
        "fraud",
        "unauthorised",
        "unauthorized",
        "stolen",
        "compromised",
        "attack",
        "problem",
        "failed",
        "failure",
        "emergency",
    ],

    "credential_request": [
        "password",
        "passcode",
        "pin",
        "username",
        "login",
        "log in",
        "verify",
        "verification",
        "account details",
        "personal details",
        "security code",
        "security question",
        "user id",
    ],
}


def keyword_score(text, keywords):
    text = text.lower()

    matches = []

    for keyword in keywords:
        if keyword in text:
            matches.append(keyword)

    return len(matches), matches


def main():

    print("=" * 70)
    print("SCAMSHIELD - WEAK CLASS CANDIDATE SELECTOR")
    print("=" * 70)

    candidates = pd.read_csv(CANDIDATES_PATH)
    human = pd.read_csv(HUMAN_PATH)

    print(f"\nFused candidates: {len(candidates)}")
    print(f"Human annotations: {len(human)}")

    human_ids = set(
        pd.to_numeric(
            human["id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
    )

    candidates["id"] = pd.to_numeric(
        candidates["id"],
        errors="coerce"
    )

    candidates = candidates[
        ~candidates["id"].isin(human_ids)
    ].copy()

    candidates = candidates.dropna(
        subset=["message"]
    )

    print(
        f"Unannotated candidates: {len(candidates)}"
    )

    records = []

    for _, row in candidates.iterrows():

        message = str(row["message"])

        for tactic, limit in TARGETS.items():

            score, matches = keyword_score(
                message,
                KEYWORDS[tactic]
            )

            if score == 0:
                continue

            records.append({
                "id": int(row["id"]),
                "message": message,
                "target_tactic": tactic,
                "keyword_score": score,
                "matched_keywords": " | ".join(matches),
                "fusion_score": float(
                    row.get("fusion_score", 0) or 0
                ),
            })

    queue = pd.DataFrame(records)

    if queue.empty:
        print("\nNo candidates found.")
        return

    # Prefer candidates with more relevant evidence,
    # while retaining some diversity.
    queue = queue.sort_values(
        ["target_tactic", "keyword_score", "fusion_score"],
        ascending=[True, False, False]
    )

    selected = []

    for tactic, limit in TARGETS.items():

        subset = queue[
            queue["target_tactic"] == tactic
        ].copy()

        # Remove duplicate messages.
        subset = subset.drop_duplicates(
            subset=["message"]
        )

        selected.append(
            subset.head(limit)
        )

    result = pd.concat(
        selected,
        ignore_index=True
    )

    result = result.sort_values(
        ["target_tactic", "keyword_score"],
        ascending=[True, False]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("TARGETED ANNOTATION QUEUE")
    print("=" * 70)

    print(
        f"\nSelected candidates: {len(result)}"
    )

    print("\nCandidates by tactic:")

    print(
        result["target_tactic"].value_counts()
    )

    for tactic in TARGETS:

        print("\n" + "-" * 70)
        print(tactic.upper())
        print("-" * 70)

        subset = result[
            result["target_tactic"] == tactic
        ]

        if subset.empty:
            print("No candidates found.")
            continue

        print(
            subset[
                [
                    "id",
                    "keyword_score",
                    "matched_keywords",
                    "message",
                ]
            ].to_string(index=False)
        )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("WEAK CLASS SELECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()