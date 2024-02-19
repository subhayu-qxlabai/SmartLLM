import re
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from models.inputs import StepsInput
from models.outputs import StepsOutput


MODEL_PATH = "/workspace/mistral_instruct_grid_search_1000epochs_4batchsize_lr1e-05/checkpoint-600"

finetuned_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/workspace/axolotl/examples/mistral/Mistral-7b-example/axolotl/src/out_llm2")


def get_prompt(inp_data):
    prompt = f"<<SYS>>  <<SYS>> [INST] {inp_data} [/INST] "
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

def get_steps(steps_input: StepsInput):
    prompt = get_prompt(steps_input.model_dump_json())
    response = get_llm_response(prompt)
    assistant_response = re.findall(r"<\|assistant\|>\n?(.*)(?:</s>)", response, flags=re.DOTALL)
    if assistant_response:
        response = assistant_response[0]
        try:
            return StepsOutput.model_validate_json(response)
        except Exception as e:
            print(e)
            try:
                return StepsOutput.model_validate_json(json.loads(response))
            except Exception as e:
                print(e)
            return response
    else:
        return response 

if __name__ == "__main__":
    input_data = {"query":"Where does Mark Zuckerberg live and what's the temperature there?","steps":["SEARCH: Find out where Mark Zuckerberg lives","RESEARCH: Determine the current temperature in that location"],"functions":[{"name":"FIND_LOCATION","description":"Locates a geographical position or address based on provided search criteria.","parameters":{"type":"object","properties":[{"name":"search_term","type":"string","description":"The term used to find a specific location or address."},{"name":"precision","type":"string","description":"The desired precision of the location search, such as 'city', 'street', or 'exact'."}],"required":["search_term"]}},{"name":"LOCATION_FINDER","description":"Locates and provides information about a specific place or address.","parameters":{"type":"object","properties":[{"name":"query","type":"string","description":"The search query, which can be a place name, address, or type of location, such as 'restaurant' or 'hospital'."},{"name":"coordinates","type":"object","description":"The geographic coordinates (latitude and longitude) to use as a reference point for the search."},{"name":"radius","type":"number","description":"The search radius in meters around the reference point within which to locate places."}],"required":["query"]}},{"name":"FORBES_DATA_RETRIEVAL","description":"Fetches data from Forbes, such as lists of the wealthiest individuals, top companies, etc.","parameters":{"type":"object","properties":[{"name":"list_type","type":"string","description":"The type of Forbes list to retrieve, e.g., 'billionaires', 'companies', 'celebrities'."},{"name":"year","type":"integer","description":"The year for which to retrieve the Forbes list."}],"required":["list_type"]}},{"name":"GENERATE_CLIMATE_REPORT","description":"Generates a detailed climate report for a specific region including temperature trends, precipitation, and extreme weather events.","parameters":{"type":"object","properties":[{"name":"region","type":"string","description":"The geographical region for which the climate report is to be generated."},{"name":"time_frame","type":"string","description":"The time frame for the climate report, such as 'current', 'historical', or a specific date range."}],"required":["region"]}},{"name":"WEATHER_FORECAST_RETRIEVAL","description":"Retrieves current weather conditions and forecasts for a specified location.","parameters":{"type":"object","properties":[{"name":"location","type":"string","description":"The location for which to retrieve weather information, such as a city name or coordinates."},{"name":"forecast_days","type":"integer","description":"The number of days to retrieve the forecast for."},{"name":"units","type":"string","description":"The units in which to return the weather data, such as 'metric' or 'imperial'. The default is 'metric'."}],"required":["location"]}},{"name":"ENERGY_PRODUCTION_ANALYSIS","description":"Analyzes energy production data from various sources to determine trends and efficiency.","parameters":{"type":"object","properties":[{"name":"energy_type","type":"string","description":"The type of energy source to analyze, such as 'solar', 'wind', 'fossil_fuel'."},{"name":"region","type":"string","description":"The geographic region for which to analyze energy production."},{"name":"time_period","type":"string","description":"The time period over which energy production data should be analyzed."}],"required":["energy_type"]}}]}
    input_schema = StepsInput.model_validate(input_data)
    output = get_steps(input_schema)
    print(output)