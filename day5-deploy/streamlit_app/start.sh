#!/bin/bash
set -e

echo "==> Starting Streamlit on port 8080..."
export PYTHONPATH=./pypackages:$PYTHONPATH
exec python3 -m streamlit run app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false