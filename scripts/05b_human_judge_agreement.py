"""
scripts/05b_human_judge_agreement.py - Calculates Cohen's Kappa between Human and LLM Judge.
"""
import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.agent import SupportAgent
from src.evaluator import SupportEvaluator

def main():
    print("=== Step 4: Human-Judge Agreement (Cohen's Kappa) ===")
    
    csv_path = config.PROCESSED_DATA_DIR / "golden_set_labeled.csv"
    if not csv_path.exists():
        print("Missing hand-labeled golden set.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path).dropna(subset=['true_intent', 'reference_criteria'])
    
    # Take a random sample of 10 cases to establish a baseline agreement metric
    sample_df = df.sample(n=10, random_state=42)
    
    agent = SupportAgent()
    evaluator = SupportEvaluator()
    
    llm_scores = []
    human_scores = []
    
    print("\nYou will be shown 10 drafted replies.")
    print("For each, enter a 'Correctness' score from 1 to 5 (1=Terrible, 5=Perfect).")
    print("-" * 50)
    
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        q = row["customer_query"]
        true_intent = row["true_intent"]
        ref = row["reference_criteria"]
        
        # Get Agent Reply
        p_agent = agent.process_case(q)
        reply = p_agent["drafted_reply"]
        
        # Get LLM Judge Score
        judge = evaluator.judge_reply(q, true_intent, reply, ref)
        llm_score = judge.get("correctness", 3)
        llm_scores.append(llm_score)
        
        # Get Human Score
        print(f"\n[Case {i}/10]")
        print(f"Query:    {q}")
        print(f"Criteria: {ref}")
        print(f"Reply:    {reply}")
        
        while True:
            try:
                human_input = int(input("Your Correctness Score (1-5): "))
                if 1 <= human_input <= 5:
                    human_scores.append(human_input)
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Enter a number between 1 and 5.")

    print("\n" + "="*50)
    print("Agreement Calculation Complete")
    print("="*50)
    print(f"LLM Scores:   {llm_scores}")
    print(f"Human Scores: {human_scores}")
    
    kappa = cohen_kappa_score(human_scores, llm_scores, weights='quadratic')
    
    print(f"\nCohen's Kappa (Quadratic Weighted): {kappa:.3f}")
    if kappa < 0: print("Interpretation: Poor agreement (worse than random)")
    elif kappa < 0.20: print("Interpretation: Slight agreement")
    elif kappa < 0.40: print("Interpretation: Fair agreement")
    elif kappa < 0.60: print("Interpretation: Moderate agreement")
    elif kappa < 0.80: print("Interpretation: Substantial agreement")
    else: print("Interpretation: Almost perfect agreement")

if __name__ == "__main__":
    main()
