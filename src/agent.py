"""
src/agent.py - The production RAG-based support triage agent.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from src.llm_client import CachedLLMClient
from src.taxonomy import TAXONOMY, INTENT_NAMES

class SupportAgent:
    def __init__(self):
        self.llm = CachedLLMClient()
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.classifier_model = config.MODELS["classifier"]
        self.generator_model = config.MODELS["generator"]
        
        # Load Retrieval Pool
        self.retrieval_df = pd.read_json(config.PROCESSED_DATA_DIR / "retrieval_pool.jsonl", lines=True)
        print(f"Agent loaded {len(self.retrieval_df)} historical cases for RAG.")
        
        # Precompute dense embeddings for the entire retrieval pool
        self.pool_embeddings = self.embedder.encode(
            self.retrieval_df["customer_query"].tolist(), 
            normalize_embeddings=True
        )

        # Escalation Policy Parameters
        # Cost of False Auto-Handle (Brand damage, regulatory, customer churn) = $50
        # Cost of False Escalation (Wasted human agent time) = $5
        # Threshold calculation: Escalate if P(failure) > Cost(FE) / (Cost(FE) + Cost(FAH)) => 5 / 55 ~= 0.09
        self.risk_threshold = 0.09 

        # Hard Escalation Triggers (legal, safety, high-dollar refunds)
        self.hard_triggers_regex = re.compile(
            r"\b(lawyer|sue|court|attorney|police|illegal|fraud|scam|stolen|theft)\b", 
            flags=re.IGNORECASE
        )

    def _clean_json(self, text: str) -> dict:
        """Strips markdown formatting from LLM JSON responses."""
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        try:
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            return {}

    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Few-shot prompt for intent classification."""
        taxonomy_prompt = "\n".join([f"- {k}: {v['description']}" for k, v in TAXONOMY.items()])
        
        system_msg = (
            "You are an intent classification routing agent for @AmazonHelp. "
            "Analyze the customer query and classify it into exactly one of the provided intents. "
            "You must respond in valid JSON format: {\"predicted_intent\": \"INTENT_NAME\", \"confidence\": 0.0-1.0}\n\n"
            f"Available Intents:\n{taxonomy_prompt}"
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Customer Query: {query}\n\nOutput JSON only."}
        ]
        
        response = self.llm.generate(
            model=self.classifier_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            prompt_version="classifier_v1.0.0"
        )
        
        parsed = self._clean_json(response["content"])
        intent = parsed.get("predicted_intent", "ORDER_TRACKING_AND_DELIVERY_STATUS")
        if intent not in INTENT_NAMES:
            intent = "ORDER_TRACKING_AND_DELIVERY_STATUS"
            
        confidence = float(parsed.get("confidence", 0.5))
        return intent, confidence

    def retrieve_exemplars(self, query: str, top_k: int = 3) -> Tuple[List[Dict], float]:
        """Retrieves top-k historical resolved cases using cosine similarity."""
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(self.pool_embeddings, query_emb)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_score = float(similarities[top_indices[0]])
        
        exemplars = []
        for idx in top_indices:
            row = self.retrieval_df.iloc[idx]
            exemplars.append({
                "case_id": row["case_id"],
                "inbound": row["customer_query"],
                "outbound_reply": row["first_brand_reply"],
                "similarity": float(similarities[idx])
            })
            
        return exemplars, top_score

    def generate_reply(self, query: str, intent: str, exemplars: List[Dict]) -> Dict:
        """Drafts a grounded reply and flags fabrications."""
        exemplars_str = "\n".join([
            f"Case {ex['case_id']} | User: {ex['inbound']} | Brand: {ex['outbound_reply']}" 
            for ex in exemplars
        ])
        
        system_msg = (
            "You are an @AmazonHelp customer support agent. "
            "Draft a reply to the customer's query. You MUST ground your reply in the historical exemplars provided. "
            "Do not invent policies. If you have to invent a detail not in the exemplars (e.g., a specific link or phone number), "
            "you must flag 'fabricated_details': true.\n\n"
            "You must respond in valid JSON format: "
            "{\"drafted_reply\": \"...\", \"cited_case_ids\": [\"case_123\"], \"fabricated_details\": false}"
        )
        
        user_msg = (
            f"Customer Query: {query}\n"
            f"Predicted Intent: {intent}\n\n"
            f"Historical Resolved Cases:\n{exemplars_str}\n\n"
            "Output JSON only."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        response = self.llm.generate(
            model=self.generator_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
            prompt_version="generator_v1.0.0",
            max_tokens=2048
        )
        
        return self._clean_json(response["content"])

    def evaluate_escalation(self, query: str, confidence: float, max_sim: float, has_fabrication: bool) -> Tuple[bool, str]:
        """Calculates risk vs threshold based on cost ratios."""
        # 1. Hard Triggers
        if self.hard_triggers_regex.search(query):
            return True, "Hard trigger matched (Legal/Safety/Fraud)"
            
        if has_fabrication:
            return True, "Generator flagged hallucinated/fabricated details"

        # 2. Probabilistic Risk Thresholding
        # Risk is a blend of low classifier confidence and poor RAG retrieval similarity
        risk_score = ((1.0 - confidence) * 0.5) + ((1.0 - max_sim) * 0.5)
        
        if risk_score > self.risk_threshold:
            return True, f"Combined risk score ({risk_score:.2f}) exceeds cost threshold ({self.risk_threshold})"
            
        return False, "Risk within bounds; safe to auto-handle"

    def process_case(self, query: str) -> Dict[str, Any]:
        """End-to-end processing pipeline."""
        # 1. Classify
        intent, conf = self.classify_intent(query)
        
        # 2. Retrieve
        exemplars, max_sim = self.retrieve_exemplars(query)
        
        # 3. Generate
        gen_output = self.generate_reply(query, intent, exemplars)
        drafted_reply = gen_output.get("drafted_reply", "Please DM us for help.")
        cited_ids = gen_output.get("cited_case_ids", [])
        has_fabrication = gen_output.get("fabricated_details", True) # Default true for safety
        
        # 4. Escalate
        escalate, reason = self.evaluate_escalation(query, conf, max_sim, has_fabrication)
        
        return {
            "query": query,
            "predicted_intent": intent,
            "intent_confidence": conf,
            "retrieved_cases": [ex["case_id"] for ex in exemplars],
            "max_retrieval_similarity": max_sim,
            "drafted_reply": drafted_reply,
            "cited_case_ids": cited_ids,
            "fabricated_details": has_fabrication,
            "escalate": escalate,
            "escalation_reason": reason,
            "prompt_versions": {
                "classifier": "classifier_v1.0.0",
                "generator": "generator_v1.0.0"
            }
        }