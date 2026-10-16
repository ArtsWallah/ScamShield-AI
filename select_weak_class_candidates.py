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