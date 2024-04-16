import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY", "005dc3bfa158d830000000002")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "K005NTUEFXRIi2mMPBKFj0EFWbXMGmA")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "s3.us-east-005.backblazeb2.com")


AZURE_ACCESS_KEY = os.getenv("ACCESS_KEY", "+I84EbAL+Sc/s6fs7ZvGZ56rzOa944xERnLqEfsaZSiRnfuLidJeE66cYAbXYRzMbua/HTxc6aWB+AStXJa6KA==")
AZURE_CONNECTION_STR = os.getenv("AWS_CONNECTION_STR", "DefaultEndpointsProtocol=https;AccountName=smartllmstorage;AccountKey=+I84EbAL+Sc/s6fs7ZvGZ56rzOa944xERnLqEfsaZSiRnfuLidJeE66cYAbXYRzMbua/HTxc6aWB+AStXJa6KA==;EndpointSuffix=core.windows.net")

DATASET_REPO = "subhayu-qxlabai/SmartLLM"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN", "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")

SUPPORTED_LANGUAGES = {'afrikaans', 'akan', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese', 'azerbaijani', 'bambara', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'burmese', 'cantonese chinese', 'catalan', 'catalán', 'cebuano', 'chichewa', 'chinese', 'chinese(zh-cn)', 'chinese(zh-tw)', 'corsican', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'finnish', 'fon', 'french', 'frisian', 'galician', 'ganda', 'georgian', 'german', 'greek', 'guarani', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kashmiri', 'kazakh', 'khmer', 'kikuyu', 'kinyarwanda', 'kirundi', 'konkani', 'korean', 'kurdish', 'kyrgyz', 'lao', 'latin', 'latvian', 'lingala', 'lithuanian', 'luxembourgish', 'maithili', 'malagasy', 'malay', 'malayalam', 'mandarin chinese', 'manipuri', 'maori', 'marathi', 'mongolian', 'nepali', 'norwegian', 'odia', 'oriya', 'panjabi', 'pashto', 'pedi', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'sanskrit', 'santali', 'sesotho', 'shona', 'sindhi', 'slovak', 'slovenian', 'somali', 'southern sotho', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'tsonga', 'tswana', 'tumbuka', 'turkish', 'twi', 'urdu', 'vietnamese', 'wolof', 'xhosa', 'yoruba', 'zulu'}
