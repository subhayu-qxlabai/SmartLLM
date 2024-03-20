import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")

DATASET_REPO = "subhayu-qxlabai/SmartLLM"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
