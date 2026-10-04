from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

KEYWORD_PATH = (
    ROOT
    / "data"
    / "processed"
    / "annotation_candidates_v2.csv"
)

SEMANTIC_PATH = (
    ROOT
    / "data"
    / "processed"
    / "semantic_candidates.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fused_candidates.csv"
)


def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return series * 0

    return (series - minimum) / (maximum - minimum)


def main():

    print("=" * 65)
    print("SCAMSHIELD - CANDIDATE FUSION")
    print("=" * 65)

    keyword_df = pd.read_csv(KEYWORD_PATH)
    semantic_df = pd.read_csv(SEMANTIC_PATH)

    print(f"\nKeyword candidates: {len(keyword_df)}")
    print(f"Semantic candidates: {len(semantic_df)}")

    # --------------------------------------------------------
    # Normalize lexical scores
    # --------------------------------------------------------

    keyword_df["lexical_score_normalized"] = (
        keyword_df
        .groupby("tactic")["candidate_score"]
        .transform(min_max_normalize)
    )

    # --------------------------------------------------------
    # Normalize semantic scores
    # --------------------------------------------------------

    semantic_df["semantic_score_normalized"] = (
        semantic_df
        .groupby("tactic")["semantic_similarity"]
        .transform(min_max_normalize)
    )

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    keyword_df = keyword_df[
        [
            "id",
            "message",
            "tactic",
            "candidate_score",
            "lexical_score_normalized",
            "matched_keywords",
        ]
    ]

    semantic_df = semantic_df[
        [
            "id",
            "message",
            "tactic",
            "semantic_similarity",
            "semantic_score_normalized",
        ]
    ]

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = pd.merge(
        keyword_df,
        semantic_df,
        on=["id", "message", "tactic"],
        how="outer"
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    merged["lexical_score_normalized"] = (
        merged["lexical_score_normalized"]
        .fillna(0)
    )

    merged["semantic_score_normalized"] = (
        merged["semantic_score_normalized"]
        .fillna(0)
    )

    merged["candidate_score"] = (
        merged["candidate_score"]
        .fillna(0)
    )

    merged["semantic_similarity"] = (
        merged["semantic_similarity"]
        .fillna(0)
    )

    merged["matched_keywords"] = (
        merged["matched_keywords"]
        .fillna("")
    )

    # --------------------------------------------------------
    # Evidence detection
    # --------------------------------------------------------

    merged["lexical_present"] = (
        merged["lexical_score_normalized"] > 0
    )

    merged["semantic_present"] = (
        merged["semantic_score_normalized"] > 0
    )

    merged["evidence_type"] = "none"

    merged.loc[
        merged["lexical_present"]
        & merged["semantic_present"],
        "evidence_type"
    ] = "dual"

    merged.loc[
        merged["lexical_present"]
        & ~merged["semantic_present"],
        "evidence_type"
    ] = "lexical_only"

    merged.loc[
        ~merged["lexical_present"]
        & merged["semantic_present"],
        "evidence_type"
    ] = "semantic_only"

    # --------------------------------------------------------
    # Fusion score
    # --------------------------------------------------------

    merged["fusion_score"] = (
        0.5 * merged["lexical_score_normalized"]
        +
        0.5 * merged["semantic_score_normalized"]
    )

    # --------------------------------------------------------
    # Evidence confidence adjustment
    # --------------------------------------------------------

    merged.loc[
        merged["evidence_type"] == "lexical_only",
        "fusion_score"
    ] *= 0.75

    merged.loc[
        merged["evidence_type"] == "semantic_only",
        "fusion_score"
    ] *= 0.50

    # Dual evidence keeps full score.

    # --------------------------------------------------------
    # Rank within tactic
    # --------------------------------------------------------

    merged["fusion_rank"] = (
        merged
        .groupby("tactic")["fusion_score"]
        .rank(
            ascending=False,
            method="first"
        )
    )

    merged["human_label"] = ""

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    merged = merged.sort_values(
        ["tactic", "fusion_score"],
        ascending=[True, False]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved fused candidates to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nTotal fused candidates: "
        f"{len(merged)}"
    )

    print("\nEvidence distribution:")

    print(
        merged["evidence_type"]
        .value_counts()
    )


if __name__ == "__main__":
    main()