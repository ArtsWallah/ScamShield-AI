from pathlib import Path
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher


ROOT = Path(__file__).resolve().parents[2]

FUSED_PATH = ROOT / "data" / "processed" / "fused_candidates.csv"
HUMAN_PATH = ROOT / "data" / "processed" / "human_annotations.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "expanded_training_candidates.csv"


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


# Do NOT automatically expand these classes.
# There is not enough human-positive evidence yet.
CONSERVATIVE = {
    "threat",
    "fear",
}


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\+?\d[\d\s\-]{6,}",
        " PHONE ",
        text
    )

    text = re.sub(
        r"[^a-z0-9£$€\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokens(text):
    return set(
        re.findall(
            r"[a-z]+",
            normalize(text)
        )
    )


def jaccard(a, b):

    a = tokens(a)
    b = tokens(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def sequence(a, b):

    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def similarity(a, b):

    return (
        0.70 * jaccard(a, b)
        +
        0.30 * sequence(a, b)
    )


def human_positive_examples(human, tactic):

    column = f"label_{tactic}"

    if column not in human.columns:
        return []

    values = pd.to_numeric(
        human[column],
        errors="coerce"
    )

    return (
        human.loc[
            values == 1,
            "message"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


def main():

    print("=" * 70)
    print("SCAMSHIELD - CONTROLLED LABEL EXPANSION V5")
    print("=" * 70)

    fused = pd.read_csv(FUSED_PATH)
    human = pd.read_csv(HUMAN_PATH)

    print(f"\nFused candidates: {len(fused)}")
    print(f"Human annotations: {len(human)}")

    # ---------------------------------------------------------
    # HUMAN POSITIVE POOLS
    # ---------------------------------------------------------

    positive_examples = {}

    print("\nHuman positive examples:")

    for tactic in TACTICS:

        examples = human_positive_examples(
            human,
            tactic
        )

        positive_examples[tactic] = examples

        print(
            f"{tactic:<25}"
            f"{len(examples)}"
        )

    # ---------------------------------------------------------
    # HUMAN IDs
    # ---------------------------------------------------------

    human_ids = set(
        pd.to_numeric(
            human["id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
    )

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    rows = []

    expansion_counts = {
        tactic: 0
        for tactic in TACTICS
    }

    # ---------------------------------------------------------
    # PROCESS CANDIDATES
    # ---------------------------------------------------------

    for _, candidate in fused.iterrows():

        candidate_id = candidate["id"]

        try:
            candidate_id = int(candidate_id)
        except:
            continue

        message = candidate.get(
            "message",
            ""
        )

        if not message:
            continue

        # Never expand human-labelled records.
        if candidate_id in human_ids:
            continue

        candidate_tactic = str(
            candidate.get(
                "tactic",
                ""
            )
        ).strip()

        if candidate_tactic not in TACTICS:
            continue

        # -----------------------------------------------------
        # Conservative classes
        # -----------------------------------------------------

        if candidate_tactic in CONSERVATIVE:
            continue

        examples = positive_examples[
            candidate_tactic
        ]

        # Need at least 3 human positives.
        if len(examples) < 3:
            continue

        # -----------------------------------------------------
        # Compare candidate against HUMAN examples
        # -----------------------------------------------------

        scores = [
            similarity(
                message,
                example
            )
            for example in examples
        ]

        scores.sort(
            reverse=True
        )

        best = scores[0]

        second = (
            scores[1]
            if len(scores) > 1
            else 0
        )

        third = (
            scores[2]
            if len(scores) > 2
            else 0
        )

        # -----------------------------------------------------
        # We require multiple supporting examples.
        #
        # This prevents one accidental similar sentence from
        # creating a pseudo-label.
        # -----------------------------------------------------

        supporting_examples = sum(
            score >= 0.45
            for score in scores
        )

        # -----------------------------------------------------
        # Strong expansion condition
        # -----------------------------------------------------

        strong_match = (
            best >= 0.62
            and second >= 0.48
            and supporting_examples >= 2
        )

        # -----------------------------------------------------
        # Very strong single match
        # -----------------------------------------------------

        exceptional_match = (
            best >= 0.72
            and supporting_examples >= 1
        )

        if not (
            strong_match
            or exceptional_match
        ):
            continue

        # -----------------------------------------------------
        # Additional evidence
        # -----------------------------------------------------

        lexical = float(
            candidate.get(
                "lexical_score_normalized",
                0
            )
            or 0
        )

        semantic = float(
            candidate.get(
                "semantic_score_normalized",
                0
            )
            or 0
        )

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        confidence = (
            0.50 * best
            +
            0.25 * second
            +
            0.15 * lexical
            +
            0.10 * semantic
        )

        confidence = min(
            confidence,
            1.0
        )

        # Final safety threshold
        if confidence < 0.60:
            continue

        rows.append({
            "id": candidate_id,
            "message": message,
            "tactic": candidate_tactic,
            "similarity_best": best,
            "similarity_second": second,
            "supporting_examples": supporting_examples,
            "lexical_score": lexical,
            "semantic_score": semantic,
            "pseudo_confidence": confidence,
        })

        expansion_counts[
            candidate_tactic
        ] += 1

    # ---------------------------------------------------------
    # DATAFRAME
    # ---------------------------------------------------------

    expanded = pd.DataFrame(rows)

    if len(expanded) > 0:

        expanded = expanded.sort_values(
            [
                "tactic",
                "pseudo_confidence"
            ],
            ascending=[
                True,
                False
            ]
        )

        expanded = expanded.drop_duplicates(
            subset=[
                "id",
                "tactic"
            ]
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    expanded.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPANSION RESULTS")
    print("=" * 70)

    print(
        f"\nNew pseudo-label candidates: "
        f"{len(expanded)}"
    )

    print("\nExpanded by tactic:")

    for tactic in TACTICS:

        print(
            f"{tactic:<25}"
            f"{expansion_counts[tactic]}"
        )

    if len(expanded) > 0:

        print("\nConfidence distribution:")

        print(
            expanded[
                "pseudo_confidence"
            ].describe()
        )

        print(
            "\nTop candidates:"
        )

        print(
            expanded[
                [
                    "id",
                    "message",
                    "tactic",
                    "similarity_best",
                    "similarity_second",
                    "pseudo_confidence",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    else:

        print(
            "\nNo candidates passed the conservative "
            "expansion criteria."
        )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("CONTROLLED LABEL EXPANSION V5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()