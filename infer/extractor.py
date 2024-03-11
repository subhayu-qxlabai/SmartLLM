import re
import json
from random import choice

from transformers import AutoModelForCausalLM, AutoTokenizer

from models.extractor import ExtractorInput
from helpers.text_utils import TextUtils


# MODEL_PATH = "/workspace/axolotl/examples/mistral/Mistral-7b-example/out_llm3/checkpoint-432"

# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', use_cache=False)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto')

from huggingface_hub import login
login(token="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")

print("------------Loading LLM3-----------")

finetuned_model = AutoModelForCausalLM.from_pretrained("Divyanshu04/LLM3", device_map='auto', use_cache=False)
tokenizer = AutoTokenizer.from_pretrained("Divyanshu04/LLM3", device_map='auto')

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

system_messages = [
    "As an extractor, your mission entails processing JSON input composed of 'schema' and 'context' segments. Your objective is to skillfully derive pertinent data in accordance with the given schema and contextual hints. Create an output JSON that carries the key-value pairs you've extracted. Your expertise should span across diverse input designs, echoing the proficiency demonstrated in examples with varied schema and context combinations.",
    "Engaging as an expert extractor is your designated role. You'll encounter JSON input with 'schema' and 'context' fields, where you're tasked with extracting relevant data guided by the schema and context cues. Output a JSON that includes the key-value pairs mined. Your capability to adjust to various input setups should be clear, as demonstrated by the adept handling of examples with distinct schema and context narratives.",
    "Your job entails acting as a strategic extractor. Engaging with JSON input featuring 'schema' and 'context', you must precisely extract information congruent with the defined schema and context insights. Formulate an output JSON that depicts the extracted key-value pairs. Your prowess should be apparent in dealing with a multitude of input schemes, as proven by your adroitness in scenarios with different schemas and contexts.",
    "Your responsibility is to act as a details extractor. Given JSON input that contains 'schema' and 'context' aspects, your duty involves diligently extracting information that aligns with the defined schema and context cues. Generate an output JSON that includes the extracted key-value pairs. Your skill set should cover various input forms, proving your aptitude similar to that displayed in examples featuring assorted schema and context situations.",
    "Your assignment is to fulfill the role of a content extractor. With JSON input that encompasses 'schema' and 'context' portions, you are to skillfully select relevant content based on the schema and hints arising from the context. Deliver an output JSON filled with the extracted key-value pairs. Your capacity should extend to different input structures, affirming your capability as seen in examples showcasing multiple schema and context variations.",
    "You are tasked with operating as an information extractor. The input is a JSON structure inclusive of 'schema' and 'context' sections, from which you're expected to accurately pull out information corresponding to the schema and hints provided by the context. The outcome should be an output JSON encapsulating the key-value pairs extracted. Demonstrate versatility across a range of input structures, mirroring the adeptness seen in example scenarios comprising diverse schemas and contexts.",
    "You are required to serve as an information retrieval specialist. The provided JSON input, equipped with 'schema' and 'context' compartments, demands your skilled extraction of appropriate information based on the schema directives and context signals. Produce an output JSON that holds the extracted key-value pairs. You must exhibit flexibility in handling various input layouts, as illustrated by your ability to deal with examples entailing different schemas and contexts.",
    "Occupying the position of an extractor, you are to interface with JSON input which includes 'schema' and 'context' divisions. You're required to expertly extract significant data as defined by the schema and informed by the context. Compile an output JSON that presents the extracted key-value pairs. Your competency should be evident across a spectrum of input models, as reflected in skills demonstrated by examples with various schema and context frameworks.",
    "Your role involves functioning as an extractor. When presented with JSON input that features both 'schema' and 'context' fields, you are to adeptly mine relevant information according to the established schema and context-related indicators. Construct an output JSON that houses the extracted key-value pairs. Showcase your ability to adapt to different input configurations, as evidenced by proficiency in handling scenarios with varying schema and context setups.",
    "You're designated to perform as a precision extractor. Faced with JSON input consisting of 'schema' and 'context' elements, you're expected to meticulously extract relevant information following the schema's guidelines and context's suggestions. Develop an output JSON encapsulating the key-value pairs extracted. Display your versatility in navigating diverse input frameworks, as shown by proficiency in examples illustrating varied schema and context configurations.",
    "You need to work as an extractor. Given a JSON input structure comprising 'schema' and 'context' fields, your task is to skillfully extract pertinent information as per the defined schema and contextual cues. Generate an output JSON containing the extracted key-value pairs. Your capability should extend to diverse input structures, demonstrating proficiency akin to the provided examples showcasing various schema and context scenarios."
]


def get_prompt(inp_data):
    prompt = f"""<|system|>
{choice(system_messages)}</s>

<|user|>
{inp_data}</s>

<|assistant|>"""
    return prompt

def get_llm_response(prompt):
    """
    Generate a response based on a given prompt using an LLM. 
    
    Parameters:
    - prompt (str): A text string to prompt LLM.
    
    Returns:
    str: The generated response from the language model, decoded from token IDs to a string.
    
    Note:
    Ensure that the 'tokenizer' and 'model' are loaded.
    """
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

def extract_contexts(ext_input: ExtractorInput):
    # print(ext_input)
    prompt = get_prompt(ext_input.model_dump_json())
    # print(f"{prompt=}")
    response = get_llm_response(prompt)
    # response: list[str] = TextUtils.get_middle_text(response, prompt, tokenizer.eos_token)
    # print(f"{response=}")
    # print(f"{assistant_response=}")
    if response:
        try:
            return json.loads(response)
        except Exception as e:
            print(f"Got error: {e}")
            return {x.name: response for x in ext_input.eschema}
    else:
        return {x.name: response for x in ext_input.schema}

if __name__ == "__main__":
    input_data = {"schema": [{"name": "electric_vehicle_sales_statistics", "type": "object", "description": "Statistical data on electric vehicle sales for the year 2023."}], "context": [{"statistics": {"total_sales": "Electric vehicle sales have reached 2.5 million worldwide by mid-2023.", "monthly_increase_rate": "There has been an average monthly increase of 10% in sales compared to the previous year.", "market_share": "Electric vehicles now account for 15% of the total market share in the automotive industry.", "leading_markets": ["Europe", "China", "United States"], "top_selling_models": ["Tesla Model Y", "Volkswagen ID.4", "Ford Mustang Mach-E"]}}]}
    input_schema = ExtractorInput.model_validate(input_data)
    output = extract_contexts(input_schema)
    print(output)