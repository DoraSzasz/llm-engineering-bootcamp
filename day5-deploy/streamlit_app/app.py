"""
app.py — Streamlit chat UI for AWS App Runner deployment.

Reads API_URL from an environment variable (set in App Runner service config)
so the public URL is never committed to git.

For local testing, use app_local.py instead.
"""

import os
import streamlit as st
import requests


# ------------------------------------------------------------------
# CONFIG — API_URL must be set in App Runner's environment variables
# ------------------------------------------------------------------
API_URL = os.environ.get("API_URL")

if not API_URL:
    st.error(
        "❌ API_URL environment variable is not set. "
        "In App Runner: Service → Configuration → Environment variables "
        "→ add `API_URL` with your full API Gateway invoke URL."
    )
    st.stop()


# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Mistral Chat",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Mistral Chat")
st.caption("Mistral-7B-Instruct-v0.3 deployed on AWS SageMaker.")


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
with st.expander("💡 Try these prompts"):
    st.markdown(
        "- *Explain quantum entanglement in simple terms.*\n"
        "- *Write a haiku about debugging code.*\n"
        "- *What is deep learning?*\n"
        "- *Summarize: [paste any paragraph]*"
    )

prompt = st.text_area(
    "Your prompt",
    height=120,
    placeholder="What do you want to ask?",
)

col1, col2 = st.columns([1, 3])
with col1:
    generate = st.button("Generate", type="primary")
with col2:
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)


# ------------------------------------------------------------------
# HANDLER
# ------------------------------------------------------------------
if generate and prompt:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={"prompt": prompt, "temperature": temperature},
                timeout=60,
            )
            data = response.json()

            if response.status_code == 200:
                st.success("Response:")
                st.write(data["generated_text"])
            else:
                st.error(
                    f"Error {response.status_code}: "
                    f"{data.get('error', 'Unknown error')}"
                )

        except requests.exceptions.Timeout:
            st.error(
                "Request timed out. The endpoint may be cold — try again "
                "in 30 seconds, or verify it's `InService` in the SageMaker console."
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "Couldn't reach the API. Verify API_URL is correct and "
                "API Gateway is deployed in the same region as your Lambda."
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.divider()
st.caption(
    "🚀 Hosted on AWS App Runner · Backend on SageMaker"
)
