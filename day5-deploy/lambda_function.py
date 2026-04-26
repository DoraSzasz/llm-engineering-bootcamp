"""
lambda_function.py — Bridges API Gateway to the SageMaker inference endpoint.

Setup in AWS Console:
  - Lambda → Create function
  - Runtime: Python 3.12
  - Configuration → General → Edit timeout: 1 min 0 sec
  - Configuration → Permissions → Click execution role:
      → Add Permissions → Attach policies → AmazonSageMakerFullAccess
  - Paste this code into the Code tab → Deploy
  - Configuration → Environment variables:
      Key:   ENDPOINT_NAME
      Value: mistral-base-v1   (or whatever Step 7 of the notebook printed)
"""

import json
import os
import boto3

# Runtime client created outside the handler — reused across warm invocations
runtime = boto3.client("sagemaker-runtime")

# Read endpoint name from env var so you can swap endpoints without redeploying
ENDPOINT = os.environ["ENDPOINT_NAME"]


def lambda_handler(event, context):
    try:
        # API Gateway wraps the request body as a JSON string
        body = json.loads(event["body"])
        prompt = body["prompt"]

        # Mistral-7B-Instruct's trained prompt format
        wrapped = f"<s>[INST] {prompt} [/INST]"

        payload = {
            "inputs": wrapped,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.03,
                "return_full_text": False,
            },
        }

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())
        generated = result[0]["generated_text"]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",   # allow Streamlit on App Runner to call
            },
            "body": json.dumps({"generated_text": generated}),
        }

    except Exception as e:
        # Logged to CloudWatch for debugging
        print(f"ERROR: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
