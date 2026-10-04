from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "manipulation_annotations.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "semantic_candidates.csv"
)

SCAM_EMBEDDING_PATH = (
    ROOT
    / "data"
    / "features"
    / "scam_embeddings.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"


TACTIC_PROTOTYPES = {

    "authority":
        "A message claiming to represent a bank, government agency, "
        "police department, company, official organization or authority.",

    "urgency":
        "A message pressuring the recipient to act immediately, "
        "quickly, today, or before a deadline.",

    "threat":
        "A message threatening account suspension, penalties, fines, "
        "legal action, arrest, cancellation or other negative consequences.",

    "fear":
        "A message attempting to create fear or anxiety by warning "
        "about fraud, hacking, security problems, stolen information "
        "or dangerous consequences.",

    "impersonation":
        "A message pretending to be a trusted person, bank, company, "
        "government organization, police officer or official service.",

    "credential_request":
        "A message asking for passwords, PINs, OTPs, verification codes, "
        "login information, security codes or authentication credentials.",

    "payment_request":
        "A message requesting money, payment, transfer, deposit, fees "
        "or financial transactions.",

    "reward_lure":
        "A message enticing the recipient with prizes, winnings, rewards, "
        "free offers, bonuses, cash or gifts.",
}


def main():

    print("=" * 65)
    print("SCAMSHIELD - SEMANTIC CANDIDATE RETRIEVAL")
    print("=" * 65)

    # --------------------------------------------------------
    # Load scam dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    messages = df["message"].astype(str).tolist()

    print(f"\nLoaded {len(messages)} scam messages.")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Sentence Transformer...")

    encoder = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Generate embeddings specifically for scam messages
    # --------------------------------------------------------

    if SCAM_EMBEDDING_PATH.exists():

        print(
            "\nLoading cached scam embeddings..."
        )

        message_embeddings = np.load(
            SCAM_EMBEDDING_PATH
        )

        print(
            f"Embedding shape: "
            f"{message_embeddings.shape}"
        )

        # Safety check
        if len(message_embeddings) != len(messages):

            print(
                "\nCached embeddings do not match "
                "the current dataset."
            )

            print(
                "Regenerating embeddings..."
            )

            message_embeddings = encoder.encode(
                messages,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            np.save(
                SCAM_EMBEDDING_PATH,
                message_embeddings
            )

    else:

        print(
            "\nGenerating embeddings for "
            "653 scam messages..."
        )

        message_embeddings = encoder.encode(
            messages,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        np.save(
            SCAM_EMBEDDING_PATH,
            message_embeddings
        )

        print(
            "\nSaved scam embeddings to:"
        )

        print(
            SCAM_EMBEDDING_PATH
        )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if len(message_embeddings) != len(df):

        raise ValueError(
            f"Embedding/data mismatch: "
            f"{len(message_embeddings)} embeddings "
            f"vs {len(df)} messages."
        )

    # --------------------------------------------------------
    # Encode tactic prototypes
    # --------------------------------------------------------

    prototype_names = list(
        TACTIC_PROTOTYPES.keys()
    )

    prototype_texts = list(
        TACTIC_PROTOTYPES.values()
    )

    print(
        "\nEncoding tactic prototypes..."
    )

    prototype_embeddings = encoder.encode(
        prototype_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Calculate semantic similarity
    # --------------------------------------------------------

    print(
        "\nCalculating semantic similarity..."
    )

    similarity_matrix = cosine_similarity(
        message_embeddings,
        prototype_embeddings
    )

    results = []

    # --------------------------------------------------------
    # Retrieve top candidates
    # --------------------------------------------------------

    for tactic_index, tactic in enumerate(
        prototype_names
    ):

        similarities = similarity_matrix[
            :,
            tactic_index
        ]

        top_indices = np.argsort(
            similarities
        )[::-1][:30]

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            results.append({
                "id": df.iloc[index]["id"],
                "message": messages[index],
                "tactic": tactic,
                "semantic_similarity": round(
                    float(similarities[index]),
                    4
                ),
                "semantic_rank": rank,
                "human_label": "",
            })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "tactic",
            "semantic_rank"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nGenerated "
        f"{len(results_df)} semantic candidates."
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Display top examples
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "TOP SEMANTIC CANDIDATES"
    )

    print(
        "=" * 65
    )

    for tactic in prototype_names:

        print(
            f"\n--- {tactic.upper()} ---"
        )

        subset = results_df[
            results_df["tactic"] == tactic
        ].head(3)

        for _, row in subset.iterrows():

            print(
                f"\nSimilarity: "
                f"{row['semantic_similarity']}"
            )

            print(
                f"Message: "
                f"{row['message']}"
            )


if __name__ == "__main__":
    main()