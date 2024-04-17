import json
from pathlib import Path

directory_path = Path("eval_dataset_for_evaluation_metric")
questions_file_path = directory_path / "questions.json"
split_file_path = directory_path / "gold_standards.json"


with open(questions_file_path, "r+") as que_file:
    dict_json = json.load(que_file)
    del dict_json[135]
    que_file.seek(0)
    json.dump(dict_json, que_file)
    que_file.truncate()

# Reading and updating split_file_path
with open(split_file_path, "r+") as split_file:
    split_json = json.load(split_file)
    del split_json[135]
    split_file.seek(0)
    json.dump(split_json, split_file)
    split_file.truncate()
