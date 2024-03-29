import re
import json
from random import choice

from transformers import AutoModelForCausalLM, AutoTokenizer

from models.generic import QuestionSplit

from helpers.text_utils import TextUtils
from pathlib import Path



# MODEL_PATH = "/workspace/models/out_llm1/checkpoint-3680"

# finetuned_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("/workspace/models/out_llm1/",device_map="auto")

from huggingface_hub import login
login(token="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")

finetuned_model = AutoModelForCausalLM.from_pretrained("vipinkatara/mLLM1_model", device_map='auto', use_cache=False)
tokenizer = AutoTokenizer.from_pretrained("vipinkatara/mLLM1_model", device_map='auto')


output_schema = QuestionSplit.schema_json()

# system_messages = [
#     'Your duty as a sincere and intelligent assistant adept at translating questions into feasible steps entails evaluating a question received in JSON form. The output should be a JSON object with two primary elements: can_i_answer (designated as true if the question can be resolved with the tools at hand, or false if it demands external resources) and tasks, which is a roster of actions for answering the question through external means, should can_i_answer turn out false.',
#     'Operating as an honest and clever assistant with the ability to deconstruct questions into straightforward steps, you are tasked to process a question presented in JSON format. Your response should be structured as a JSON object containing two keys: can_i_answer (true if the question can be addressed without external resources, or false otherwise), and tasks, a sequence of steps that detail how to answer the query using external sources, should can_i_answer be set to false.',
#     'Taking on the responsibility as an honest and astute assistant skilled in simplifying queries into digestible steps, you are presented with a JSON input question. You are obligated to supply a JSON object as a return, carrying the keys can_i_answer (true insinuates that the question can be answered with no need for external resources, or false if otherwise) and tasks, listing necessary steps for answering the question with external help if can_i_answer is established as false.',
#     'Your role as an upright and sharp-witted assistant who can distill questions into simple procedural steps involves handling an input question in JSON. You must deliver a JSON object in response, equipped with the keys can_i_answer (indicating true if the question is answerable with the resources at hand, or false if not) and tasks, outlining the process for answering the question with external resources if can_i_answer is false.',
# ]

system_messages = [
    "Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false."
]


# def get_prompt(_input):
#     if isinstance(_input, str):
#         _input = {"question": _input}
#     prompt_template = f"""### Instruction: {choice(system_messages)} Your output should have the following schema: {output_schema}

# ### Input: 
# {_input}
        
# ### Response: 
# """
#     prompt_template="<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {\"question\": \"What is the latest news on india\"} [/INST] "
#     return prompt_template

def get_prompt(_input):
    if isinstance(_input, str):
        _input = {"question": _input}
        # _input=json.dumps(_input).strip()
        # prompt_template = f"""[INST] {_input} [/INST] ### Response: {{question:{_input},tasks:[string,string],can_i_answer:string}}"""
        prompt_template = f"""<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {_input} [/INST] """
        # prompt_template="<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {\"question\": \"What is the latest news on india\"} [/INST] "
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
    
    return decoded_output[0]

# -----------------------------------
# TODO: Remove this block afterwards

import json
from models.generic import QuestionSplit, BaseModel
from pydantic import root_validator

class SplitOutput(BaseModel):
    output: QuestionSplit
    
    @root_validator(pre=True)
    def validate(cls, values: dict):
        values["output"] = json.loads(values["output"])
        return values
    
# ----------------------------------

def break_question(question: str):
    prompt = get_prompt(question)

    response = get_llm_response(prompt)
    
    response = TextUtils.get_middle_text(response, prompt, tokenizer.eos_token).strip()
    if response:
        try:
            return SplitOutput.model_validate_json(response)
        except Exception as e:
            print(e)
            return response
    else:
        return response 




# if __name__ == "__main__":
#     query = "जयपुर, भारत में देखने लायक प्रसिद्ध चीज़ें क्या हैं?"
#     response = break_question(query)
#     print(response)