"""
app_local.py — Streamlit chat UI for local laptop testing.

Identical to app.py, but with clearer error messages tailored to local dev
and a friendly "running locally" footer.

Usage:
  cd streamlit_app
  python3 -m venv venv
  source venv/bin/activate           # Windows: venv\\Scripts\\activate
  pip install -r requirements.txt
  
  export API_URL="https://YOUR_API_GATEWAY_URL/invoke"
  streamlit run app_local.py

Browser opens to http://localhost:8501.
"""

import os
import streamlit as st
import requests


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
API_URL = os.environ.get("API_URL")

if not API_URL:
    st.error(
        "❌ API_URL environment variable is not set.\n\n"
        "Before running locally:\n"
        "```bash\n"
        "export API_URL='https://YOUR_API_GATEWAY_URL/invoke'\n"
        "streamlit run app_local.py\n"
        "```"
    )
    st.stop()


# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Mistral Chat (local)",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Mistral Chat")
st.caption("Mistral-7B-Instruct-v0.3 deployed on AWS SageMaker · running locally on your laptop")


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
                "Request timed out (60s). Possible causes:\n"
                "- The SageMaker endpoint is cold — try again in 30s\n"
                "- Your laptop has no internet connection\n"
                "- Corporate VPN is blocking AWS API endpoints"
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "Couldn't reach the API. Possible causes:\n"
                "- API_URL is wrong (check the value vs your API Gateway invoke URL)\n"
                "- API Gateway and Lambda are in different regions\n"
                "- API Gateway hasn't been deployed yet (check Stages → $default)"
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.divider()
st.caption(
    f"💻 Running locally · Backend: `{API_URL.split('/invoke')[0]}`"
)
