#!/usr/bin/env bash
# Run the LLM Validator app — PHI-safe, fully offline
#
# Telemetry is disabled via:
#   1. .streamlit/config.toml (gatherUsageStats = false)
#   2. Environment variable below (belt-and-suspenders)
#   3. server.address = localhost (never exposed to network)

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ADDRESS=localhost

# Verify streamlit is installed before starting
if ! python3 -c "import streamlit" 2>/dev/null; then
  echo "Streamlit not found. Install dependencies first:"
  echo "  pip install -r requirements.txt --no-index --find-links ./wheels/"
  echo "  (or with internet access: pip install -r requirements.txt)"
  exit 1
fi

python3 -m streamlit run app.py \
  --browser.gatherUsageStats false \
  --server.address localhost \
  --server.port 8501
