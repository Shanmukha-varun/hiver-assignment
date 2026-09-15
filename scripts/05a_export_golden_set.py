"""
scripts/05a_export_golden_set.py - Samples 200 cases (stratified + hard cases) for manual labeling.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.baselines import SimpleBaseline

def main():
    print("=== Exporting Golden Set for Manual Labeling ===")
    
    # Load holdout pool and baseline model
    eval_pool = pd.read_json(config.PROCESSED_DATA_DIR / "eval_pool.jsonl", lines=True)
    simple = SimpleBaseline.load(config.PROCESSED_DATA_DIR / "simple_baseline.pkl")
    
    print("Scoring holdout pool to find stratified boundaries and hard cases...")
    
    # Get rough predictions to guide sampling
    preds = [simple.predict(q) for q in eval_pool["customer_query"]]
    eval_pool["rough_intent"] = [p["predicted_intent"] for p in preds]
    eval_pool["confidence"] = [p["intent_confidence"] for p in preds]
    
    # 1. Stratified Sampling (~20 per intent)
    stratified = eval_pool.groupby("rough_intent").apply(
        lambda x: x.sample(min(len(x), 20), random_state=42)
    ).reset_index(drop=True)
    
    # 2. Hard-Case Sampling (40 cases with the absolute lowest confidence)
    remaining = eval_pool[~eval_pool["case_id"].isin(stratified["case_id"])]
    hard_cases = remaining.sort_values("confidence").head(40)
    
    # Combine and shuffle
    golden_set = pd.concat([stratified, hard_cases]).sample(frac=1.0, random_state=42)
    
    # Format for human labeling
    export_df = golden_set[["case_id", "customer_query"]].copy()
    export_df["true_intent"] = ""
    export_df["true_escalate"] = ""
    export_df["reference_criteria"] = ""
    
    output_path = config.PROCESSED_DATA_DIR / "golden_set_unlabeled.csv"
    export_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Exported {len(export_df)} cases to {output_path}")
    print("Next Steps:")
    print("1. Open the CSV in Excel/Google Sheets.")
    print("2. Fill in 'true_intent' (must match exact taxonomy names).")
    print("3. Fill in 'true_escalate' (TRUE or FALSE).")
    print("4. Fill in 'reference_criteria' (e.g., 'Apologize and ask for order number in DM').")
    print("5. Save the completed file as 'golden_set_labeled.csv' in the same folder.")

if __name__ == "__main__":
    main()