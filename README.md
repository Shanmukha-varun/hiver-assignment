# Hiver AI Support Triage Agent: @AmazonHelp

This repository implements a production-lean AI support triage agent for `@AmazonHelp`, built using the `thoughtvector/customer-support-on-twitter` dataset. The system handles data-driven intent classification, RAG-grounded reply generation, and cost-weighted escalation routing.

This submission prioritizes evaluation rigor, deterministic fallback mechanisms, and honest metric reporting over raw model size.

---

## Runnable Pipeline & Reproducibility

The pipeline is heavily cached to disk using SHA-256 hashes of the request payloads. **Evaluating the system takes less than 10 seconds and costs 0 API credits.**

**Prerequisites:** Python 3.10+
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add raw data
# Place twcs.csv inside data/raw/

# 3. Configure environment variables (Free-tier API Keys)
# Create a .env file in the root directory:
GROQ_API_KEY="your_groq_key"
