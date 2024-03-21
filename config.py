import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY", "005dc3bfa158d830000000002")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "K005NTUEFXRIi2mMPBKFj0EFWbXMGmA")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "s3.us-east-005.backblazeb2.com")

DATASET_REPO = "subhayu-qxlabai/SmartLLM"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN", "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")
