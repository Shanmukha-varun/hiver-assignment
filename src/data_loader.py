"""
src/data_loader.py - Reconstructs conversation threads, cleans tweet text,
and creates structured case units for retrieval and evaluation.
"""
import re
import pandas as pd
from typing import Dict, List, Optional
import config

class TWCSDataLoader:
    def __init__(self, raw_csv_path: str = str(config.RAW_DATA_PATH), brand: str = config.TARGET_BRAND):
        self.raw_csv_path = raw_csv_path
        self.brand = brand

    @staticmethod
    def clean_tweet_text(text: str, target_brand: str) -> str:
        """
        Cleans tweet noise while preserving affective cues (emoji, caps, punctuation):
        - Strips routing @mentions (@AmazonHelp, @115888)
        - Normalizes short links to canonical placeholder
        - Collapses repeated whitespace
        """
        if not isinstance(text, str):
            return ""

        # Remove brand mention and anonymous user mentions
        text = re.sub(rf"@{target_brand}\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"@\d+\b", "", text)
        text = re.sub(r"@[A-Za-z0-9_]+\b", "", text)

        # Normalize tracking/shortened URLs while tagging domain if present
        text = re.sub(r"https?://(?:amzn\.to|t\.co)/\S+", "[LINK]", text)
        text = re.sub(r"https?://\S+", "[LINK]", text)

        # Normalize spaces while keeping punctuation and emoji intact
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", " ", text).strip()
        return text

    def load_and_reconstruct(self, max_turns: int = config.MAX_TURNS_PER_CASE) -> pd.DataFrame:
        """
        Filters TWCS dataset for target brand and reconstructs multi-turn cases.
        """
        print(f"Loading TWCS raw data from: {self.raw_csv_path}")
        df = pd.read_csv(self.raw_csv_path, low_memory=False)

        # Normalize IDs to string
        df["tweet_id"] = df["tweet_id"].astype(str)
        df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)

        # Identify all brand interactions
        brand_tweets = df[df["author_id"] == self.brand]
        brand_tweet_ids = set(brand_tweets["tweet_id"])

        # Filter inbound tweets addressed to brand
        inbound_mask = df["inbound"] & df["text"].str.contains(rf"@{self.brand}\b", case=False, na=False)
        inbound_tweets = df[inbound_mask]

        print(f"Found {len(brand_tweets)} outbound tweets from {self.brand}")
        print(f"Found {len(inbound_tweets)} inbound tweets mentioning @{self.brand}")

        # Index lookup maps
        tweet_lookup = df.set_index("tweet_id").to_dict(orient="index")

        # Find initial customer inquiries (inbound tweets that are NOT in response to another tweet)
        initial_inbounds = inbound_tweets[inbound_tweets["in_response_to_tweet_id"] == ""]

        cases: List[Dict] = []

        for _, row in initial_inbounds.iterrows():
            first_tweet_id = row["tweet_id"]
            first_text_raw = row["text"]
            first_text_clean = self.clean_tweet_text(first_text_raw, self.brand)

            if len(first_text_clean) < 10:
                continue

            # Thread reconstruction
            thread = [{
                "turn": 1,
                "speaker": "customer",
                "tweet_id": first_tweet_id,
                "text_raw": first_text_raw,
                "text_clean": first_text_clean,
                "created_at": row.get("created_at", "")
            }]

            curr_id = first_tweet_id
            turns_collected = 1

            # Traverse forward by looking at responses
            while turns_collected < max_turns:
                # Find tweet that directly responds to curr_id
                # Check response_tweet_id from current tweet if available
                curr_node = tweet_lookup.get(curr_id)
                if not curr_node:
                    break

                response_ids_str = str(curr_node.get("response_tweet_id", ""))
                if not response_ids_str or response_ids_str == "nan":
                    break

                next_ids = [tid.strip().replace(".0", "") for tid in response_ids_str.split(",") if tid.strip()]
                if not next_ids:
                    break

                next_id = next_ids[0]
                next_tweet = tweet_lookup.get(next_id)
                if not next_tweet:
                    break

                speaker = "brand" if next_tweet["author_id"] == self.brand else "customer"
                turns_collected += 1
                thread.append({
                    "turn": turns_collected,
                    "speaker": speaker,
                    "tweet_id": next_id,
                    "text_raw": next_tweet["text"],
                    "text_clean": self.clean_tweet_text(next_tweet["text"], self.brand),
                    "created_at": next_tweet.get("created_at", "")
                })
                curr_id = next_id

            # Filter for cases that have at least one brand reply
            brand_replies = [t for t in thread if t["speaker"] == "brand"]
            if not brand_replies:
                continue

            first_brand_reply = brand_replies[0]["text_clean"]

            cases.append({
                "case_id": f"case_{first_tweet_id}",
                "root_tweet_id": first_tweet_id,
                "customer_query_raw": first_text_raw,
                "customer_query": first_text_clean,
                "first_brand_reply": first_brand_reply,
                "turn_count": len(thread),
                "full_thread": thread,
            })

        cases_df = pd.DataFrame(cases)
        print(f"Reconstructed {len(cases_df)} valid resolution cases for {self.brand}.")
        return cases_df

    def split_and_save(
        self,
        cases_df: pd.DataFrame,
        retrieval_size: int = config.RETRIEVAL_POOL_SIZE,
        eval_size: int = config.EVAL_POOL_SIZE,
        seed: int = 42
    ) -> None:
        """
        Splits cases into a historical knowledge pool for RAG grounding
        and an unobserved evaluation pool for testing and golden set sampling.
        """
        shuffled = cases_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        retrieval_pool = shuffled.iloc[:retrieval_size].copy()
        eval_pool = shuffled.iloc[retrieval_size:retrieval_size + eval_size].copy()

        retrieval_pool_path = config.PROCESSED_DATA_DIR / "retrieval_pool.jsonl"
        eval_pool_path = config.PROCESSED_DATA_DIR / "eval_pool.jsonl"

        retrieval_pool.to_json(retrieval_pool_path, orient="records", lines=True)
        eval_pool.to_json(eval_pool_path, orient="records", lines=True)

        print(f"Saved {len(retrieval_pool)} cases to {retrieval_pool_path}")
        print(f"Saved {len(eval_pool)} cases to {eval_pool_path}")