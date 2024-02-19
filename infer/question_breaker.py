import re
import json
from random import choice

from transformers import AutoModelForCausalLM, AutoTokenizer

from models.generic import QuestionSplit


MODEL_PATH = "/workspace/mistral_instruct_grid_search_llm2_1000epochs_4batchsize_lr1e-05/checkpoint-1000"

finetuned_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

output_schema = QuestionSplit.schema_json()

system_messages = [
    'Your duty as a sincere and intelligent assistant adept at translating questions into feasible steps entails evaluating a question received in JSON form. The output should be a JSON object with two primary elements: can_i_answer (designated as true if the question can be resolved with the tools at hand, or false if it demands external resources) and tasks, which is a roster of actions for answering the question through external means, should can_i_answer turn out false.',
    'Operating as an honest and clever assistant with the ability to deconstruct questions into straightforward steps, you are tasked to process a question presented in JSON format. Your response should be structured as a JSON object containing two keys: can_i_answer (true if the question can be addressed without external resources, or false otherwise), and tasks, a sequence of steps that detail how to answer the query using external sources, should can_i_answer be set to false.',
    'Taking on the responsibility as an honest and astute assistant skilled in simplifying queries into digestible steps, you are presented with a JSON input question. You are obligated to supply a JSON object as a return, carrying the keys can_i_answer (true insinuates that the question can be answered with no need for external resources, or false if otherwise) and tasks, listing necessary steps for answering the question with external help if can_i_answer is established as false.',
    'Your role as an upright and sharp-witted assistant who can distill questions into simple procedural steps involves handling an input question in JSON. You must deliver a JSON object in response, equipped with the keys can_i_answer (indicating true if the question is answerable with the resources at hand, or false if not) and tasks, outlining the process for answering the question with external resources if can_i_answer is false.',
]

def get_prompt(_input):
    if isinstance(_input, str):
        _input = {"question": _input}
    prompt_template = f"""### Instruction: {choice(system_messages)} Your output should have the following schema: {output_schema}

### Input: 
{_input}
        
### Response: 
"""
    return prompt_template

def get_llm_response(prompt):
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    generated_ids = finetuned_model.generate(
        **model_inputs, 
        max_new_tokens=2048, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_output = tokenizer.batch_decode(generated_ids)
    
    return decoded_output[0]

def break_question(question: str):
    prompt = get_prompt(question)
    response = get_llm_response(prompt)
    assistant_response = re.findall(r"### Response:\s*\n?(.*)(?:</s>)", response, flags=re.DOTALL)
    if assistant_response:
        response = assistant_response[0]
        try:
            return QuestionSplit(question=question, **json.loads(response))
        except Exception as e:
            print(e)
            return response
    else:
        return response 


if __name__ == "__main__":
    query = "Who is the prime minister of India?"
    response = break_question(query)
    print(response)