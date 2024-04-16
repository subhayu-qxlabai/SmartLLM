import re
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "bigscience/bloom-7b1"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="auto")

def ask_llm(question: str):
    messages = [
        {"role": "user", "content": question},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    
    model_inputs = encodeds.to(device)
    # model.to(device)
    
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    response = re.findall(r"\[\/INST\]\s*(.*)<\/s>", decoded[0], re.DOTALL) or [decoded[0]]
    return response[0]
    