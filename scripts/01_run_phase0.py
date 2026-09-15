"""
scripts/01_run_phase0.py - Runner script for Phase 0.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import TWCSDataLoader

def main():
    print("=== Running Phase 0: Data Setup & Thread Reconstruction ===")
    loader = TWCSDataLoader(raw_csv_path=str(config.RAW_DATA_PATH), brand=config.TARGET_BRAND)
    
    cases_df = loader.load_and_reconstruct(max_turns=config.MAX_TURNS_PER_CASE)
    loader.split_and_save(cases_df, retrieval_size=config.RETRIEVAL_POOL_SIZE, eval_size=config.EVAL_POOL_SIZE)
    print("=== Phase 0 Complete ===")

if __name__ == "__main__":
    main()