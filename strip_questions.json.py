
import json

from pathlib import Path

directory_path = Path("eval_dataset_for_evaluation_metric")
questions_file_path = directory_path / "questions.json"


# Step 1: Open the JSON file and load its contents
with open(questions_file_path, 'r') as file:
    data = json.load(file)

# Step 2: Strip each string in the list
stripped_data = [string.strip() for string in data]

# Step 3: Save the stripped strings back into a JSON file
with open(questions_file_path, 'w') as file:
    file.seek(0)
    json.dump(stripped_data, file, indent=4)
