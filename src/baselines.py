"""
src/baselines.py - Trivial and Simple baseline models for @AmazonHelp.
"""
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import config
from src.taxonomy import INTENT_NAMES

class TrivialBaseline:
    """
    Trivial Baseline:
    1. Intent: Always predicts the majority intent ('ORDER_TRACKING_AND_DELIVERY_STATUS').
    2. Generation: Returns a single canned auto-response.
    3. Escalation: Fixed policy (always AUTO_HANDLE to highlight risk of missed escalations).
    """
    def __init__(self, majority_intent: str = "ORDER_TRACKING_AND_DELIVERY_STATUS"):
        self.majority_intent = majority_intent
        self.canned_reply = (
            "Thanks for reaching out to Amazon Help. We apologize for the inconvenience. "
            "Please check your order status on your account page or DM us with your order details."
        )

    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.majority_intent,
            "intent_confidence": 1.0,
            "drafted_reply": self.canned_reply,
            "escalate": False,
            "escalation_reason": "Trivial baseline policy: always auto-handle",
            "model_version": "trivial_baseline_v1",
        }


class SimpleBaseline:
    """
    Simple Baseline:
    1. Intent: TF-IDF (word + char n-grams) + Multinomial Logistic Regression.
    2. Retrieval/Reply: Exact lookup of the single most frequent historical brand reply
       for that predicted intent in the training set (no LLM generation).
    3. Escalation: Regular expression keyword rule covering high-risk triggers.
    """
    ESCALATION_KEYWORDS = [
        r"\blawyer\b",
        r"\battorney\b",
        r"\bsue\b",
        r"\blegal\b",
        r"\bpolice\b",
        r"\bcourt\b",
        r"\bfraud\b",
        r"\bstolen\b",
        r"\btheft\b",
        r"\bdispute\b",
        r"\bchargeback\b",
        r"\bunauthorized\b",
        r"\bhacked\b",
        r"\bconsumer protection\b",
        r"\btrading standards\b",
        r"\bbetter business bureau\b",
        r"\bbbb\b",
        r"\brefund now\b",
    ]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.intent_to_canned_reply: Dict[str, str] = {}
        self.keyword_regex = re.compile("|".join(self.ESCALATION_KEYWORDS), flags=re.IGNORECASE)
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str], brand_replies: List[str]):
        """Fits TF-IDF + Logistic Regression and caches most frequent historical reply per intent."""
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

        # Build empirical reply lookup per intent
        df_fit = pd.DataFrame({"label": labels, "reply": brand_replies})
        for intent in INTENT_NAMES:
            subset = df_fit[df_fit["label"] == intent]
            if not subset.empty and subset["reply"].str.len().gt(15).any():
                common_reply = Counter(subset["reply"]).most_common(1)[0][0]
                self.intent_to_canned_reply[intent] = common_reply
            else:
                self.intent_to_canned_reply[intent] = (
                    "Please contact our support team with your order ID so we can investigate."
                )

        self.is_fitted = True

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("SimpleBaseline must be fitted before predict()")

        # 1. Intent Classification
        X = self.vectorizer.transform([text])
        pred_intent = self.clf.predict(X)[0]
        probs = self.clf.predict_proba(X)[0]
        confidence = float(max(probs))

        # 2. Historical Reply Retrieval
        drafted_reply = self.intent_to_canned_reply.get(
            pred_intent,
            "Please send us a DM with your order details so we can assist."
        )

        # 3. Rule-based Escalation Decision
        keyword_match = self.keyword_regex.search(text)
        if keyword_match:
            escalate = True
            reason = f"Keyword escalation trigger matched: '{keyword_match.group(0)}'"
        elif confidence < 0.35:
            escalate = True
            reason = f"Low intent classification confidence ({confidence:.2f} < 0.35)"
        else:
            escalate = False
            reason = "No critical keywords detected and confidence meets baseline threshold"

        return {
            "predicted_intent": pred_intent,
            "intent_confidence": confidence,
            "drafted_reply": drafted_reply,
            "escalate": escalate,
            "escalation_reason": reason,
            "model_version": "simple_baseline_tfidf_lr_v1",
        }

    def save(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "SimpleBaseline":
        with open(filepath, "rb") as f:
            return pickle.load(f)