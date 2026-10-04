from pathlib import Path
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher


ROOT = Path(__file__).resolve().parents[2]

FUSED_PATH = ROOT / "data" / "processed" / "fused_candidates.csv"
HUMAN_PATH = ROOT / "data" / "processed" / "human_annotations.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "pseudo_labeled_tactics.csv"


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


# Classes with too little human data to safely pseudo-label.
CONSERVATIVE_TACTICS = {
    "threat",
    "fear",
}


def normalize_text(text):
    """Normalize SMS text for similarity comparison."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Replace phone numbers
    text = re.sub(r"\+?\d[\d\s\-]{6,}", " PHONE ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def token_set(text):
    """Create a simple normalized token set."""

    text = normalize_text(text)

    tokens = re.findall(
        r"[a-z]+",
        text
    )

    return set(tokens)


def jaccard_similarity(a, b):
    """Token-level similarity."""

    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(
        a_tokens.intersection(b_tokens)
    )

    union = len(
        a_tokens.union(b_tokens)
    )

    return intersection / union


def sequence_similarity(a, b):
    """Character-level similarity."""

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def similarity(a, b):
    """
    Combined similarity.

    Token overlap is weighted more heavily because
    SMS scam tactics often reuse characteristic phrases.
    """

    jac = jaccard_similarity(a, b)
    seq = sequence_similarity(a, b)

    return (
        0.70 * jac
        +
        0.30 * seq
    )


def main():

    print("=" * 70)
    print("SCAMSHIELD - TACTIC-AWARE PSEUDO LABELER V4")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    fused = pd.read_csv(FUSED_PATH)
    human = pd.read_csv(HUMAN_PATH)

    print(f"\nFused candidates: {len(fused)}")
    print(f"Human annotation records: {len(human)}")

    # ---------------------------------------------------------
    # NORMALIZE IDs
    # ---------------------------------------------------------

    fused["id"] = pd.to_numeric(
        fused["id"],
        errors="coerce"
    )

    human["id"] = pd.to_numeric(
        human["id"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # HUMAN DATA
    # ---------------------------------------------------------

    if "annotation_status" in human.columns:

        completed = human[
            human["annotation_status"]
            .astype(str)
            .str.lower()
            .eq("completed")
        ].copy()

    else:

        completed = human.copy()

    print(
        f"Completed human annotations: "
        f"{len(completed)}"
    )

    # ---------------------------------------------------------
    # CREATE OUTPUT
    # ---------------------------------------------------------

    result = fused.copy()

    for tactic in TACTICS:

        result[f"label_{tactic}"] = np.nan

    result["label_source"] = "unlabeled"
    result["human_annotated"] = False
    result["pseudo_confidence"] = 0.0

    # ---------------------------------------------------------
    # ADD HUMAN LABELS
    # ---------------------------------------------------------

    human_label_columns = [
        f"label_{tactic}"
        for tactic in TACTICS
    ]

    human_ids = set()

    for _, row in completed.iterrows():

        message_id = row["id"]

        if pd.isna(message_id):
            continue

        matches = (
            result["id"] == message_id
        )

        if not matches.any():
            continue

        human_ids.add(message_id)

        for tactic in TACTICS:

            column = f"label_{tactic}"

            if column not in completed.columns:
                continue

            value = row[column]

            if pd.isna(value):
                continue

            try:
                value = int(value)
            except (ValueError, TypeError):
                continue

            if value not in [0, 1]:
                continue

            result.loc[
                matches,
                column
            ] = value

        result.loc[
            matches,
            "human_annotated"
        ] = True

        result.loc[
            matches,
            "label_source"
        ] = "human"

    print(
        f"Human-labelled candidate IDs matched: "
        f"{len(human_ids)}"
    )

    # ---------------------------------------------------------
    # BUILD TACTIC-SPECIFIC HUMAN POSITIVE POOLS
    # ---------------------------------------------------------

    positive_examples = {}

    for tactic in TACTICS:

        column = f"label_{tactic}"

        examples = completed[
            completed[column]
            .apply(
                lambda x: str(x).strip() in ["1", "1.0"]
            )
        ]

        texts = (
            examples["message"]
            .dropna()
            .astype(str)
            .tolist()
        )

        positive_examples[tactic] = texts

        print(
            f"{tactic:<25} "
            f"human positives: {len(texts)}"
        )

    # ---------------------------------------------------------
    # PSEUDO LABELING
    # ---------------------------------------------------------

    pseudo_count = 0

    pseudo_by_tactic = {
        tactic: 0
        for tactic in TACTICS
    }

    print("\nGenerating tactic-specific pseudo-labels...")

    for index, row in result.iterrows():

        # Never overwrite human labels.
        if row["human_annotated"]:
            continue

        message = row.get("message", "")

        if not message:
            continue

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # We use the candidate's own tactic.
        # -----------------------------------------------------

        tactic = str(
            row.get("tactic", "")
        ).strip()

        if tactic not in TACTICS:
            continue

        # -----------------------------------------------------
        # Conservative classes
        # -----------------------------------------------------

        if tactic in CONSERVATIVE_TACTICS:
            continue

        examples = positive_examples.get(
            tactic,
            []
        )

        # Need at least 3 human positive examples
        # before generating pseudo-labels.
        if len(examples) < 3:
            continue

        # -----------------------------------------------------
        # Find maximum similarity to any human positive
        # -----------------------------------------------------

        similarities = []

        for example in examples:

            score = similarity(
                message,
                example
            )

            similarities.append(score)

        if not similarities:
            continue

        best_similarity = max(
            similarities
        )

        # -----------------------------------------------------
        # Additional evidence from fusion pipeline
        # -----------------------------------------------------

        lexical = float(
            row.get(
                "lexical_score_normalized",
                0
            )
            or 0
        )

        semantic = float(
            row.get(
                "semantic_score_normalized",
                0
            )
            or 0
        )

        evidence = str(
            row.get(
                "evidence_type",
                ""
            )
        )

        # -----------------------------------------------------
        # Conservative pseudo-label thresholds
        #
        # Strong human similarity:
        #       >= 0.65
        #
        # OR moderate human similarity plus strong
        # semantic/lexical evidence.
        # -----------------------------------------------------

        direct_match = (
            best_similarity >= 0.65
        )

        supported_match = (
            (best_similarity >= 0.50)
            and
            (
                semantic >= 0.75
                or lexical >= 0.75
            )
        )

        if not (
            direct_match
            or supported_match
        ):
            continue

        # -----------------------------------------------------
        # Final confidence
        # -----------------------------------------------------

        confidence = (
            0.60 * best_similarity
            +
            0.20 * semantic
            +
            0.20 * lexical
        )

        if evidence == "semantic_only":
            confidence *= 0.90

        elif evidence == "lexical_only":
            confidence *= 0.95

        confidence = min(
            confidence,
            1.0
        )

        # Final safety threshold
        if confidence < 0.60:
            continue

        column = f"label_{tactic}"

        result.loc[
            index,
            column
        ] = 1

        result.loc[
            index,
            "label_source"
        ] = "pseudo"

        result.loc[
            index,
            "pseudo_confidence"
        ] = confidence

        pseudo_count += 1
        pseudo_by_tactic[tactic] += 1

    # ---------------------------------------------------------
    # TRAINING ELIGIBILITY
    # ---------------------------------------------------------

    result["training_eligible"] = False

    # Human-labelled rows are always eligible.
    result.loc[
        result["human_annotated"],
        "training_eligible"
    ] = True

    # Only high-confidence pseudo-labels.
    result.loc[
        (
            result["label_source"] == "pseudo"
        )
        &
        (
            result["pseudo_confidence"] >= 0.60
        ),
        "training_eligible"
    ] = True

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL SOURCE")
    print("=" * 70)

    print(
        result["label_source"]
        .value_counts()
    )

    print(
        f"\nTotal pseudo-labels created: "
        f"{pseudo_count}"
    )

    print("\nPseudo-labels by tactic:")

    for tactic in TACTICS:

        print(
            f"{tactic:<25} "
            f"{pseudo_by_tactic[tactic]}"
        )

    print("\nTraining eligibility:")

    print(
        result["training_eligible"]
        .value_counts()
    )

    # ---------------------------------------------------------
    # TACTIC DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TACTIC DISTRIBUTION")
    print("=" * 70)

    for tactic in TACTICS:

        column = f"label_{tactic}"

        positive = (
            result[column] == 1
        ).sum()

        negative = (
            result[column] == 0
        ).sum()

        missing = (
            result[column].isna()
        ).sum()

        print(
            f"{tactic:<25}"
            f" Positive: {positive:<5}"
            f" Negative: {negative:<5}"
            f" Missing: {missing}"
        )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    pseudo_rows = result[
        result["label_source"] == "pseudo"
    ]

    print("\n" + "=" * 70)
    print("PSEUDO CONFIDENCE")
    print("=" * 70)

    if len(pseudo_rows) > 0:

        print(
            pseudo_rows[
                "pseudo_confidence"
            ].describe()
        )

    else:

        print("No pseudo-labels generated.")

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("PSEUDO-LABELING V4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()