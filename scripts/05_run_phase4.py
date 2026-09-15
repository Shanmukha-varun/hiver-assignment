"""
scripts/05_run_phase4.py - Runs all models against the HAND-LABELED golden set.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.llm_client import CachedLLMClient
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.agent import SupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.evaluator import SupportEvaluator

def load_real_golden_set() -> pd.DataFrame:
    """Loads the hand-labeled CSV file."""
    csv_path = config.PROCESSED_DATA_DIR / "golden_set_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}. You must label golden_set_unlabeled.csv first and rename it!")
    
    golden_df = pd.read_csv(csv_path)
    
    # Enforce strict boolean types for the escalate column just in case Excel messed it up
    golden_df["true_escalate"] = golden_df["true_escalate"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    
    # Forward-fill any missing criteria with a generic fallback so the judge doesn't crash
    golden_df["reference_criteria"] = golden_df["reference_criteria"].fillna("Address the customer issue politely.")
    
    return golden_df

def main():
    print("=== Running Phase 4: REAL Golden Set Evaluation Pipeline ===")
    golden_df = load_real_golden_set()
    print(f"Loaded Hand-Labeled Golden Set: {len(golden_df)} cases.")
    
    # Initialize Systems
    trivial = TrivialBaseline()
    simple = SimpleBaseline.load(config.PROCESSED_DATA_DIR / "simple_baseline.pkl")
    agent = SupportAgent()
    evaluator = SupportEvaluator()
    
    results = {"Trivial": [], "Simple": [], "Agent": []}
    
    print("Evaluating models (caching will make reruns instant)...")
    for _, row in tqdm(golden_df.iterrows(), total=len(golden_df)):
        q = row["customer_query"]
        true_intent = row["true_intent"]
        ref = row["reference_criteria"]
        
        # Run predictions
        p_triv = trivial.predict(q)
        p_simp = simple.predict(q)
        p_agent = agent.process_case(q)
        
        # Run Judge on Agent
        judge_scores = evaluator.judge_reply(q, true_intent, p_agent["drafted_reply"], ref)
        p_agent["judge_scores"] = judge_scores
        
        results["Trivial"].append(p_triv)
        results["Simple"].append(p_simp)
        results["Agent"].append(p_agent)
        
    # Calculate System-Level Metrics
    print("\n" + "="*50)
    print("                PHASE 4 EVALUATION RESULTS")
    print("="*50)
    
    for name, sys_results in results.items():
        pred_intents = [r["predicted_intent"] for r in sys_results]
        pred_escalates = [r["escalate"] for r in sys_results]
        
        metrics = evaluator.compute_deterministic_metrics(
            golden_df["true_intent"].tolist(), pred_intents,
            golden_df["true_escalate"].tolist(), pred_escalates
        )
        
        print(f"--- {name.upper()} SYSTEM ---")
        print(f" Intent Macro F1:     {metrics['intent_macro_f1']:.3f}")
        print(f" Escalation Accuracy: {metrics['escalation_accuracy']:.3f}")
        
        if name == "Agent":
            avg_c = np.mean([r["judge_scores"].get("correctness", 3) for r in sys_results])
            avg_g = np.mean([r["judge_scores"].get("groundedness", 3) for r in sys_results])
            avg_t = np.mean([r["judge_scores"].get("tone", 3) for r in sys_results])
            avg_a = np.mean([r["judge_scores"].get("actionability", 3) for r in sys_results])
            print(f" LLM Judge (1-5):     Corr: {avg_c:.1f} | Ground: {avg_g:.1f} | Tone: {avg_t:.1f} | Act: {avg_a:.1f}")
            
    print("="*50)
    CachedLLMClient.print_failure_stats()
    print("=== Phase 4 Complete ===")

if __name__ == "__main__":
    main()
