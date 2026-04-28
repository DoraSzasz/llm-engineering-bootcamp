#!/bin/bash
# start.sh — App Runner startup script for Streamlit
#
# Runs at container startup. Installs Python dependencies, then launches
# Streamlit. Used because App Runner's managed Python runtime doesn't
# preserve build-time pip installs into the runtime stage.
#
# App Runner Configuration:
#   Build command: echo skipping build
#   Start command: bash start.sh

set -e   # exit on first error

echo "==> Installing Python dependencies..."
pip3 install --user -r requirements.txt

echo "==> Starting Streamlit on port 8080..."
exec python3 -m streamlit run app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
