"""
scripts/03_run_phase2.py - Trains and serializes the Simple Baseline model.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.baselines import TrivialBaseline, SimpleBaseline
from src.taxonomy import INTENT_NAMES

# High-precision keyword anchors for bootstrap training
INTENT_RULES = {
    "ACCOUNT_ACCESS_AND_SECURITY": ["otp", "2fa", "locked", "password", "hack", "unauthorized", "sign in", "login"],
    "PACKAGE_DAMAGED_OR_MISSING": ["empty box", "damaged", "broken", "stolen", "missing", "never received", "marked delivered"],
    "SUBSCRIPTION_AND_DIGITAL_SERVICES": ["prime", "kindle", "audible", "music", "membership", "streaming", "video"],
    "REFUND_AND_RETURN_REQUEST": ["refund", "return", "drop off", "ups", "pickup", "credited", "send back"],
    "PAYMENT_AND_PROMOTIONAL_PRICING": ["charge", "double charged", "promo code", "voucher", "gift card", "declined", "discount"],
    "PRODUCT_AVAILABILITY_AND_INQUIRY": ["in stock", "restock", "when will", "seller", "availability", "buy"],
    "SERVICE_COMPLAINT_AND_AGENT_FEEDBACK": ["hung up", "rude", "worst service", "useless", "representative", "agent", "complaint"],
}

def rule_assign_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, terms in INTENT_RULES.items():
        if any(term in text_lower for term in terms):
            return intent
    return "ORDER_TRACKING_AND_DELIVERY_STATUS"

def main():
    print("=== Running Phase 2: Train Simple Baseline Pipeline ===")
    retrieval_file = config.PROCESSED_DATA_DIR / "retrieval_pool.jsonl"
    
    if not retrieval_file.exists():
        raise FileNotFoundError(f"Missing {retrieval_file}. Run Phase 0 first.")

    df = pd.read_json(retrieval_file, lines=True)
    sample_train = df.head(1500).copy()

    sample_train["intent_label"] = sample_train["customer_query"].apply(rule_assign_intent)
    
    print("\nTraining set intent distribution for Simple Baseline:")
    print(sample_train["intent_label"].value_counts())

    simple_baseline = SimpleBaseline()
    simple_baseline.fit(
        texts=sample_train["customer_query"].tolist(),
        labels=sample_train["intent_label"].tolist(),
        brand_replies=sample_train["first_brand_reply"].tolist()
    )

    baseline_model_path = config.PROCESSED_DATA_DIR / "simple_baseline.pkl"
    simple_baseline.save(baseline_model_path)
    print(f"\nSimple Baseline saved to: {baseline_model_path}")

    # Smoke test on test inputs
    trivial_baseline = TrivialBaseline()
    test_queries = [
        "Where is my package? It was supposed to be delivered yesterday.",
        "Your driver stole my package and I am calling my lawyer!",
        "Can I get a refund for my Prime membership?",
    ]

    print("\n--- Smoke Test Comparisons ---")
    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        t_out = trivial_baseline.predict(q)
        s_out = simple_baseline.predict(q)
        print(f" [Trivial] Intent: {t_out['predicted_intent']} | Escalate: {t_out['escalate']}")
        print(f" [Simple]  Intent: {s_out['predicted_intent']} ({s_out['intent_confidence']:.2f}) | Escalate: {s_out['escalate']} ({s_out['escalation_reason']})")

    print("\n=== Phase 2 Complete ===")

if __name__ == "__main__":
    main()