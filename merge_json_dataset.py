import json
from pathlib import Path
import pandas as pd
from pandas import json_normalize
import nltk
from nltk import *
from nltk.corpus import *
import time
from func_timeout import func_timeout, FunctionTimedOut
from transformers import AutoModelForCausalLM, AutoTokenizer
from helpers.text_utils import TextUtils
from openai_language_detection import language_detection
from huggingface_hub import login


login(token="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")

print("------------Loading LLM1-----------")


# finetuned_model = AutoModelForCausalLM.from_pretrained("vipinkatara/mLLM1_model", device_map='auto', use_cache=False,offload_folder=folder_offload, offload_state_dict=True)
# tokenizer = AutoTokenizer.from_pretrained("vipinkatara/mLLM1_model", device_map='auto',offload_folder=folder_offload, offload_state_dict=True)


finetuned_model = AutoModelForCausalLM.from_pretrained("vipinkatara/mLLM1_model", device_map='auto', use_cache=False)
tokenizer = AutoTokenizer.from_pretrained("vipinkatara/mLLM1_model", device_map='auto')

# nltk.download("stopwords")
 
# text = "Bonjour tout le monde"  # Example text
# language = detect(text)
# print("Detected language:", language)

system_messages = [
    "Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false."
]

def get_prompt(_input):
    if isinstance(_input, str):
        _input = {"question": _input}
        # _input=json.dumps(_input).strip()
        # prompt_template = f"""[INST] {_input} [/INST] ### Response: {{question:{_input},tasks:[string,string],can_i_answer:string}}"""
        prompt_template = f"""<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {_input} [/INST] """
        return prompt_template


def get_llm_response(prompt):
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    generated_ids = finetuned_model.generate(
        **model_inputs, 
        max_new_tokens=8096, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_output = tokenizer.batch_decode(generated_ids)
    start_string=prompt + ' {"output": "'
    end_string='"}'+tokenizer.eos_token
    
    response = TextUtils.get_middle_text(decoded_output[0], start_string, end_string).strip()
    response = response.replace('\\"', '"')
    return response

def lang_ratio(input):    
    lang_ratio={}
    tokens=wordpunct_tokenize(input)
    words=[word.lower() for word in tokens]
    for language in stopwords.fileids():
        stopwords_set=set(stopwords.words(language))
        words_set=set(words)
        common_elements=words_set.intersection(stopwords_set)
        lang_ratio[language]=len(common_elements)
    return lang_ratio
        

def language_identification(input):
    ratios = lang_ratio(input)
    language=max(ratios,key=ratios.get)
    print(language)

directory_path = Path("eval_dataset_for_evaluation_metric")
questions_file_path = directory_path / "questions.json"
split_file_path = directory_path / "gold_standards.json"


with open(questions_file_path) as que_file:
    dict_json = json.load(que_file)

with open(split_file_path,"r") as split_file:
    split_json = json.load(split_file)

ques_df=pd.DataFrame(dict_json,columns=["questions"])

def find_question(question):
    for item in split_json:
        if item["question"] == question:
            return json.dumps(item)
   
ques_df["expected_output"] = ques_df["questions"].apply(find_question)


def change_question_to_prompt_template(value):
    value = {"question": value}
    prompt_template = f"""<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {value} [/INST] """
    return prompt_template


def create_actual_output_column(value):
    try:
        actual_output=func_timeout(30,get_llm_response,args=(value,))
        print(f"Response received from LLM")
        return actual_output
    except FunctionTimedOut as e:
        print(e)
        actual_output=''
        return actual_output
    except Exception as e:
        print(e)
        actual_output=''
        return actual_output
    
def create_language_column(value):
    # print(value)
    res=language_detection(value)
    print(f"Response received from openai for language is {res}")
    return res

ques_df["language_type"]=ques_df["questions"].apply(create_language_column)

ques_df["questions"] = ques_df["questions"].apply(change_question_to_prompt_template)

ques_df['actual_output'] = ques_df['questions'].apply(create_actual_output_column)


def apply_function_with_delay(df, batch_size=100, delay_seconds=5):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch['actual_output'] = batch['questions'].apply(create_actual_output_column)
        time.sleep(delay_seconds)
        yield batch


ques_df.dropna(inplace=True)
allowed_languages = ['English', 'Spanish', 'Telugu', 'Italian', 'Hindi']
filtered_df = ques_df[ques_df['language_type'].isin(allowed_languages)]
json_data = filtered_df.to_json(orient='records')

eval_dataset_path = directory_path / "eval_dataset_LLM2.json"

with open(eval_dataset_path, 'w') as f:
    f.seek(0)
    f.write(json_data)
    f.truncate()

    
# print(df['questions'][874])
# language_identification(df['questions'][873])