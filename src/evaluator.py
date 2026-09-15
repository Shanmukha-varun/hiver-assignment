"""
src/evaluator.py - Metrics calculation and LLM-as-Judge rubric scoring.
"""
import json
from typing import Dict, List, Any
from sklearn.metrics import f1_score, accuracy_score
from src.llm_client import CachedLLMClient
import config

class SupportEvaluator:
    def __init__(self):
        self.llm = CachedLLMClient()
        self.judge_model = config.MODELS["judge"]

    def _clean_json(self, text: str) -> dict:
        clean = text.strip()
        if clean.startswith("```json"): clean = clean[7:]
        if clean.startswith("```"): clean = clean[3:]
        if clean.endswith("```"): clean = clean[:-3]
        try:
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            return {"correctness": 3, "groundedness": 3, "tone": 3, "actionability": 3}

    def compute_deterministic_metrics(self, true_intents: List[str], pred_intents: List[str], 
                                      true_escalations: List[bool], pred_escalations: List[bool]) -> Dict[str, float]:
        """Calculates Macro F1 for intents and Accuracy for escalation routing."""
        return {
            "intent_macro_f1": f1_score(true_intents, pred_intents, average="macro", zero_division=0),
            "intent_accuracy": accuracy_score(true_intents, pred_intents),
            "escalation_accuracy": accuracy_score(true_escalations, pred_escalations)
        }

    def judge_reply(self, query: str, true_intent: str, generated_reply: str, reference_criteria: str) -> Dict[str, Any]:
        """Runs Opus-equivalent (gpt-oss-120b) to score reply quality from 1-5."""
        system_msg = (
            "You are an expert QA evaluator for @AmazonHelp customer support. "
            "Score the provided drafted reply from 1 to 5 on four axes:\n"
            "1. correctness: Does it address the specific issue?\n"
            "2. groundedness: Does it avoid hallucinating policies or links not provided?\n"
            "3. tone: Is it empathetic, professional, and matching brand voice?\n"
            "4. actionability: Does it give the user clear next steps?\n\n"
            "Output JSON only: {\"correctness\": int, \"groundedness\": int, \"tone\": int, \"actionability\": int, \"rationale\": \"str\"}"
        )
        
        user_msg = (
            f"Query: {query}\n"
            f"True Intent: {true_intent}\n"
            f"Acceptability Criteria: {reference_criteria}\n\n"
            f"Drafted Reply to Evaluate: {generated_reply}\n\n"
            "Output JSON only."
        )

        response = self.llm.generate(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            prompt_version="judge_v1.0.0",
            max_tokens=512
        )
        return self._clean_json(response["content"])