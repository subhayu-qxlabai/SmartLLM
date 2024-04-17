from dataset_gen.smart_llm import question_generator, split_generator, topic_generator
import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# def add_prefix(example):
#     example["sentence1"] = 'My sentence: ' + example["sentence1"]
#     return example


def gen_que():
    directory_path = Path("eval_dataset_for_evaluation_metric")

    if not directory_path.exists():
        directory_path.mkdir()

    que_file_path = directory_path / "questions5.json"
    dataset = load_dataset("cais/mmlu","miscellaneous").shuffle(seed=42)["validation"]

    existing_data=dataset["question"][:50]
    print("-"*10 + "Append method in questions.json file started" + "-"*10)
    with open(que_file_path, 'a') as file:
        file.seek(0)
        json.dump(existing_data, file)
    print("-"*10 + "Append method in questions.json file ended" + "-"*10)
        


def gen_question_split(path:Path):
    print("-"*10 + "Question splits started" + "-"*10)
    # print(path)
    split_list=[]
    try:
         with open(path, 'r') as file:
            existing_data = json.load(file)
            split_gen = split_generator.QuestionSplitGenerator()
            chunk_size=5
            split_existing_data=[existing_data[i:i + chunk_size] for i in range(0, len(existing_data), chunk_size)]
            for small_list in split_existing_data:
                generated_splits = split_gen.generate(small_list, dump=False)
                # print(generated_splits)
                split_list.extend(generated_splits)
    except FileNotFoundError:
        print("questions.json File not found.")

    except PermissionError:
        print("Permission denied to read the questions.json file.")

    except json.JSONDecodeError:
        print("JSON decoding error or empty questions.json file.")
    except Exception as e:
        print("An error occurred:", e)

    print("-"*10 + "Question splits ended" + "-"*10)
    directory_path = Path("eval_dataset_for_evaluation_metric")

    if not directory_path.exists():
        directory_path.mkdir()

    split_file_path = directory_path / "gold_standards6.json"

    print("-"*10 + "Append method in gold_standards.json file started" + "-"*10)
    try:
        with open(split_file_path, 'a+') as file:
            file.seek(0)
            existing_data = json.load(file) if file.tell() != 0 else []
            if split_list:
                existing_data.extend(split_list)
                file.seek(0)
                file.truncate(0)  # Clear the file before writing
                json.dump(existing_data, file)
            
    except FileNotFoundError:
        print("gold_standards.json File not found.")

    except PermissionError:
        print("Permission denied to read the gold_standards.json file.")
    except Exception as e:
        print("An error occurred:", e)
    print("-"*10 + "Append method in gold_standards.json file ended" + "-"*10)


# gen_que()
directory_path = Path("eval_dataset_for_evaluation_metric")
questions_file_path = directory_path / "questions6.json"
gen_question_split(path=questions_file_path)

# directory_path = Path("eval_dataset_for_evaluation_metric")
# split_file_path = directory_path / "gold_standards4.json"
# # questions_file_path = directory_path / "questions.json"
# with open(split_file_path, 'r+') as file:
#     file.seek(0)
#     existing_data = json.load(file)
#     df=pd.DataFrame(existing_data)
#     answer_true=df[df["can_i_answer"]==True]
#     print(answer_true.count())