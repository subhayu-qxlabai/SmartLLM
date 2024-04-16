from dataset_gen.smart_llm import split_generator
from pathlib import Path
import json

directory_path = Path("eval_dataset_for_evaluation_metric")
questions_file_path = directory_path / "questions.json"

split_list=[]

with open(questions_file_path, 'r') as file:
    existing_data = json.load(file)
    split_gen = split_generator.QuestionSplitGenerator()
# split_gen=split_generator.QuestionSplitGenerator()
    
    question=existing_data[338:340]
    # print(question)
    
    generated_splits = split_gen.generate(question, dump=False)
    # print(generated_splits)
    
    split_list.extend(generated_splits)
    
split_file_path = directory_path / "gold_standards.json"
with open(split_file_path, 'r+') as file:
    file.seek(0)
    existing_data2 = json.load(file)
    # print(len(existing_data2))
    existing_data2.extend(split_list)
    print(len(existing_data2))
    file.seek(0)
    json.dump(existing_data2, file)


# directory_path = Path("eval_dataset_for_evaluation_metric")
# questions_file_path = directory_path / "questions.json"
# print(gen_question_split(path=questions_file_path))
