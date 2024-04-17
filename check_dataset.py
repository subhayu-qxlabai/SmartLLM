from dataset_gen.smart_llm import question_generator, split_generator, topic_generator
import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
import re

pattern = re.compile(r'user\s(.*?)\<\|im_end\|\>', re.DOTALL | re.UNICODE)
def extract_question(x):
    match = pattern.search(x)
    if match:
        return match.group(1).strip()
    else:
        return ''


dataset = load_dataset("Felladrin/ChatML-aya_dataset")["train"]
df=pd.DataFrame(dataset)
df=df[df["language"]=="English"][200:300]
# print(df[:70])
# extract_question(df[:50])
df['text'] = df['text'].apply(extract_question)
# print(df[df['text']!=""])
print(json.dumps(df['text'].to_list()))

