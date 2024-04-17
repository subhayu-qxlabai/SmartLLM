from dataset_gen.smart_llm import question_generator, split_generator, topic_generator
from pathlib import Path
import json


# def split_list(lst, chunk_size):
#     return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

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

    split_file_path = directory_path / "gold_standards4.json"

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
    # return split_list


def generate_evaluation_metric_dataset():
    languages=["english","hindi","spanish","telugu","italian"]
    topic_gen = topic_generator.TopicGenerator()
    print("-"*10 + "Random Task generation started" + "-"*10)
    random_topics = topic_gen.generate(n=3, dump=False)
    print("-"*10 + "Random Task generation finished" + "-"*10)
    combined_questions=[]
    question_gen = question_generator.QuestionGenerator()
    print("-"*10 + "Random Questions generation from each tasks started" + "-"*10)
    for topic in random_topics:
        for lang in languages:
            n_questions = 3
            generated_questions = question_gen.generate(topic, n=n_questions, language=lang, dump=False)
            combined_questions.extend(generated_questions)
    print("-"*10 + "Random Questions generation from each tasks finished" + "-"*10)
    directory_path = Path("eval_dataset_for_evaluation_metric")

    if not directory_path.exists():
        directory_path.mkdir()
        
    questions_file_path = directory_path / "questions4.json"
    print("-"*10 + "Append method in questions.json file started" + "-"*10)
    with open(questions_file_path, 'a') as file:
        existing_data = json.load(file) if file.tell() != 0 else []
        existing_data.extend(combined_questions)
        file.seek(0)
        json.dump(existing_data, file)
    print("-"*10 + "Append method in questions.json file ended" + "-"*10)
    gen_question_split(path=questions_file_path)


generate_evaluation_metric_dataset()

# directory_path = Path("eval_dataset_for_evaluation_metric")
# questions_file_path = directory_path / "questions.json"
# gen_question_split(path=questions_file_path)
        
    