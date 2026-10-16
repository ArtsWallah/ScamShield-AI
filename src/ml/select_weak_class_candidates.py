from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CANDIDATES_PATH = DATA_DIR / "fused_candidates.csv"
HUMAN_PATH = DATA_DIR / "human_annotations.csv"
WEAK_OUTPUT_PATH = DATA_DIR / "weak_class_annotation_queue.csv"
HARD_NEGATIVE_OUTPUT_PATH = DATA_DIR / "hard_negative_annotation_queue.csv"

# Per-tactic cap. This is a candidate cap, not a target number of positive labels.
TARGETS = {
    "threat": 30,
    "fear": 30,
    "credential_request": 30,
    "impersonation": 30,
    "authority": 30,
    "payment_request": 30,
}

KEYWORDS = {
    "threat": [
        "close", "closed", "blocked", "block", "cancel", "cancelled",
        "suspend", "suspended", "expire", "expired", "warning", "final",
        "last chance", "action required", "failure", "terminate", "termination",
        "deactivate", "deactivated", "legal action", "penalty",
    ],
    "fear": [
        "danger", "warning", "risk", "security", "fraud", "unauthorised",
        "unauthorized", "stolen", "compromised", "attack", "problem", "failed",
        "failure", "emergency", "suspicious activity", "unusual activity",
        "unauthorized access", "alert",
    ],
    "credential_request": [
        "password", "passcode", "pin", "username", "login", "log in", "verify",
        "verification", "account details", "personal details", "security code",
        "security question", "user id", "otp", "one time password", "access code",
        "confirm your identity", "sign in",
    ],
    "impersonation": [
        "bank", "customer service", "support team", "security team", "account team",
        "government", "tax office", "police", "delivery team", "paypal", "apple",
        "microsoft", "amazon", "your bank", "official notice", "service provider",
    ],
    "authority": [
        "official", "department", "administrator", "admin", "manager", "director",
        "government", "court", "police", "tax", "compliance", "regulation",
        "authorized", "authorised", "policy", "legal notice",
    ],
    "payment_request": [
        "pay", "payment", "transfer", "send money", "fee", "invoice", "bill",
        "refund", "bank transfer", "wire", "deposit", "due amount", "outstanding",
        "wallet", "transaction", "payment required", "collect payment",
    ],
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value


def keyword_matches(text: str, keywords: list[str]) -> list[str]:
    lower = str(text).lower()
    return [kw for kw in keywords if kw in lower]


def numeric_series(df: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def backup_before_replace(path: Path) -> None:
    if path.exists():
        backup = path.with_name(path.stem + ".backup" + path.suffix)
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")


def load_inputs():
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Missing candidate file: {CANDIDATES_PATH}")
    if not HUMAN_PATH.exists():
        raise FileNotFoundError(f"Missing annotation file: {HUMAN_PATH}")

    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"id": str})
    human = pd.read_csv(HUMAN_PATH, dtype={"id": str})

    for col in ("id", "message"):
        if col not in candidates.columns:
            raise ValueError(f"{CANDIDATES_PATH.name} is missing required column: {col}")
    if "id" not in human.columns:
        raise ValueError(f"{HUMAN_PATH.name} is missing required column: id")

    candidates["id"] = candidates["id"].map(normalize_id)
    human["id"] = human["id"].map(normalize_id)
    candidates = candidates[(candidates["id"] != "") & candidates["message"].notna()].copy()
    human = human[human["id"] != ""].copy()

    annotated_ids = set(human["id"])
    annotated_texts = set(
        human["message"].dropna().map(normalize_text)
    ) if "message" in human.columns else set()

    candidates["_normalized_message"] = candidates["message"].map(normalize_text)
    candidates = candidates[
        ~candidates["id"].isin(annotated_ids)
        & ~candidates["_normalized_message"].isin(annotated_texts)
        & (candidates["_normalized_message"] != "")
    ].copy()

    # Keep one candidate row per normalized message; prefer the strongest fusion score.
    candidates["_fusion_numeric"] = numeric_series(candidates, "fusion_score")
    candidates = candidates.sort_values("_fusion_numeric", ascending=False)
    candidates = candidates.drop_duplicates("_normalized_message", keep="first")
    return candidates, human


def build_rare_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in candidates.iterrows():
        message = str(row["message"])
        for tactic, limit in TARGETS.items():
            matches = keyword_matches(message, KEYWORDS[tactic])
            tactic_match = (
                "tactic" in row.index
                and str(row.get("tactic", "")).strip() == tactic
            )
            # Candidate discovery only: keyword/tactic evidence is never a label.
            if not matches and not tactic_match:
                continue
            records.append({
                "id": normalize_id(row["id"]),
                "message": message,
                "target_tactic": tactic,
                "keyword_score": len(matches),
                "matched_keywords": " | ".join(matches),
                "fusion_score": float(row.get("fusion_score", 0) or 0),
                "evidence_type": row.get("evidence_type", ""),
                "semantic_score_normalized": row.get("semantic_score_normalized", 0),
                "source_tactic": row.get("tactic", ""),
            })

    if not records:
        return pd.DataFrame(columns=[
            "id", "message", "target_tactic", "keyword_score", "matched_keywords",
            "fusion_score", "evidence_type", "semantic_score_normalized", "source_tactic",
        ])

    queue = pd.DataFrame(records)
    queue["_normalized_message"] = queue["message"].map(normalize_text)
    selected = []
    for tactic, limit in TARGETS.items():
        subset = queue[queue["target_tactic"] == tactic].copy()
        subset = subset.sort_values(
            ["keyword_score", "fusion_score"], ascending=[False, False]
        ).drop_duplicates("_normalized_message", keep="first")
        selected.append(subset.head(limit))

    result = pd.concat(selected, ignore_index=True) if selected else queue.iloc[0:0]
    return result.drop(columns=["_normalized_message"], errors="ignore")


def build_hard_negative_review_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select low-scoring candidate messages for review, without assigning negative labels."""
    if candidates.empty:
        return pd.DataFrame(columns=["id", "message", "fusion_score", "evidence_type", "review_reason"])

    frame = candidates.copy()
    frame["fusion_score"] = numeric_series(frame, "fusion_score")
    # Use the lower-scoring half of remaining fused candidates as review candidates.
    cutoff = frame["fusion_score"].quantile(0.50)
    result = frame[frame["fusion_score"] <= cutoff].copy()
    result = result.sort_values("fusion_score", ascending=True)
    columns = [
        col for col in [
            "id", "message", "tactic", "fusion_score", "evidence_type",
            "matched_keywords", "semantic_score_normalized",
        ] if col in result.columns
    ]
    result = result[columns].drop_duplicates("id", keep="first")
    result["review_reason"] = "Low-scoring candidate; verify whether tactic is actually present"
    return result


def main():
    print("=" * 72)
    print("SCAMSHIELD — EXPANDED ANNOTATION QUEUE BUILDER")
    print("=" * 72)

    candidates, human = load_inputs()
    print(f"Human annotation rows: {len(human)}")
    print(f"Unannotated unique candidate messages: {len(candidates)}")

    rare_queue = build_rare_queue(candidates)
    hard_queue = build_hard_negative_review_queue(candidates)

    backup_before_replace(WEAK_OUTPUT_PATH)
    backup_before_replace(HARD_NEGATIVE_OUTPUT_PATH)

    WEAK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rare_queue.to_csv(WEAK_OUTPUT_PATH, index=False, encoding="utf-8")
    hard_queue.to_csv(HARD_NEGATIVE_OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nRare-class queue by tactic:")
    if rare_queue.empty:
        print("No candidates found. Inspect fused_candidates.csv and its source candidate files.")
    else:
        print(rare_queue["target_tactic"].value_counts().reindex(TARGETS, fill_value=0).to_string())

    print(f"\nRare-class candidates: {len(rare_queue)}")
    print(f"Hard-negative review candidates: {len(hard_queue)}")
    print("\nSaved:")
    print(WEAK_OUTPUT_PATH)
    print(HARD_NEGATIVE_OUTPUT_PATH)
    print("\nReminder: candidates and scores are not labels. Review each message manually.")


if __name__ == "__main__":
    main()
