"""
scripts/04_run_phase3.py - Smoke test for the production RAG Agent.
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agent import SupportAgent

def main():
    print("=== Running Phase 3: Production Agent Initialization ===")
    agent = SupportAgent()
    
    test_queries = [
        "Where is my package? It was supposed to be delivered yesterday.", # Standard
        "My account got hacked and someone ordered an iPad!", # Security (Escalate)
        "What is the return policy for a defective laptop?", # Return policy (Should retrieve well)
    ]
    
    print("\n--- Agent Smoke Test ---")
    for q in test_queries:
        print(f"\n[Incoming Query]: {q}")
        result = agent.process_case(q)
        
        print(f" Intent:       {result['predicted_intent']} (Conf: {result['intent_confidence']:.2f})")
        print(f" RAG Max Sim:  {result['max_retrieval_similarity']:.2f} | Cases: {result['retrieved_cases']}")
        print(f" Draft Reply:  {result['drafted_reply']}")
        print(f" Fabrications: {result['fabricated_details']} | Cited: {result['cited_case_ids']}")
        print(f" Action:       {'ESCALATE' if result['escalate'] else 'AUTO-HANDLE'} ({result['escalation_reason']})")

    print("\n=== Phase 3 Complete ===")

if __name__ == "__main__":
    main()