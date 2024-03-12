from pathlib import Path
from typing import Callable

from models.inputs import StepsInput
from models.outputs import StepsOutput
from helpers.utils import try_json_dumps
from helpers.call_openai import call_openai_api
from dataset_gen.base.json_generator import JSONGenerator

DEFAULT_DUMP_DIR = Path("generated/step_output")

class StepOutputGenerator(JSONGenerator):
    def __init__(
        self,
        dump_dir: str | Path = DEFAULT_DUMP_DIR,
        file_prefix: str = "s",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
    ):
        system_prompt = system_prompt or (
            "You are an honest and smart assistant who can generate steps to achieve from. You don't have to actually run the functions. "
            "You just have to put them in order by using your thoughts. Your output should only be in JSON\n\n"
            'Here\'s the JSON output schema:\n{"$defs": {"ExtractEntry": {"properties": {"id": {"title": "Id", "type": "string"}, "context": {"items": {"type": "string"}, "title": "Context", "type": "array"}, "schema": {"items": {"$ref": "#/$defs/ExtractSchemaEntry"}, "title": "Schema", "type": "array"}}, "required": ["id", "context"], "title": "ExtractEntry", "type": "object"}, "ExtractSchemaEntry": {"properties": {"name": {"title": "Name", "type": "string"}, "type": {"title": "Type", "type": "string"}, "description": {"title": "Description", "type": "string"}}, "required": ["name", "type", "description"], "title": "ExtractSchemaEntry", "type": "object"}, "FunctionEntry": {"properties": {"id": {"title": "Id", "type": "string"}, "name": {"title": "Name", "type": "string"}, "parameters": {"items": {"$ref": "#/$defs/ParameterEntry"}, "title": "Parameters", "type": "array"}}, "required": ["id", "name", "parameters"], "title": "FunctionEntry", "type": "object"}, "ParameterEntry": {"properties": {"name": {"title": "Name", "type": "string"}, "value": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "number"}, {"items": {}, "type": "array"}, {"type": "object"}], "title": "Value"}}, "required": ["name", "value"], "title": "ParameterEntry", "type": "object"}, "Step": {"properties": {"id": {"title": "Id", "type": "string"}, "thought": {"title": "Thought", "type": "string"}, "explore_tools": {"title": "Explore Tools", "type": "string"}, "available_tools": {"items": {"type": "string"}, "title": "Available Tools", "type": "array"}, "choose_tool": {"title": "Choose Tool", "type": "string"}, "understand_dependencies": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Understand Dependencies"}, "extract": {"default": [], "items": {"$ref": "#/$defs/ExtractEntry"}, "title": "Extract", "type": "array"}, "function": {"items": {"$ref": "#/$defs/FunctionEntry"}, "title": "Function", "type": "array"}}, "required": ["id", "thought", "explore_tools", "available_tools", "choose_tool", "function"], "title": "Step", "type": "object"}}, "properties": {"overview": {"title": "Overview", "type": "string"}, "steps": {"items": {"$ref": "#/$defs/Step"}, "title": "Steps", "type": "array"}}, "required": ["overview", "steps"], "title": "Process", "type": "object"}'
        )
        example_messages = example_messages or [
            {
                "role": "user",
                "content": '{"query":"What is the current temperature in the country where Elon Musk resides?","steps":["Find the country where Elon Musk resides","Find the current temperature in that country"],"functions":[{"name":"GOOGLE_SEARCH","description":"Performs a Google search and retrieves information based on the provided query.","parameters":{"type":"object","properties":[{"name":"query","type":"string","description":"The search query to be executed on Google."}],"required":["query"]}},{"name":"WEATHER_INFO","description":"Retrieves weather information based on the specified location.","parameters":{"type":"object","properties":[{"name":"location","type":"string","description":"The location for which weather information is requested. For example, \'New York, NY\'."}],"required":["location"]}},{"name":"FORMAT_TEXT","description":"Formats text with placeholders and contextual information.","parameters":{"type":"object","properties":[{"name":"context","type":"string","description":"The text to be formatted, including placeholders for function parameters."},{"name":"query","type":"string","description":"The specific query to be included in the formatted text."}],"required":["context","query"]}},{"name":"WIKIPEDIA","description":"Retrieves information from Wikipedia based on the provided search query.","parameters":{"type":"object","properties":[{"name":"query","type":"string","description":"The search query used to retrieve information from Wikipedia."}],"required":["query"]}},{"name":"GENERAL_LLM","description":"Generates natural language understanding and responses based on the provided query and context.","parameters":{"type":"object","properties":[{"name":"query","type":"string","description":"The primary query or input for the language model."},{"name":"context","type":"string","description":"Contextual information to provide additional context for the language model."}],"required":["query"]}}]}',
            },
            {
                "role": "assistant",
                "content": '{"overview":"We need to determine the current temperature in Elon Musk\'s home country. To do this, we must first identify Elon Musk\'s home country and then find out the temperature there.","steps":[{"id":"step_1","thought":"First, we need to find out Elon Musk\'s home country.","explore_tools":"To determine Elon Musk\'s home country, I have several options that can assist me.","available_tools":["GOOGLE_SEARCH","GENERAL_LLM","WIKIPEDIA"],"choose_tool":"I should choose the \'GOOGLE_SEARCH\' tool to find Elon Musk\'s home country as it uses the Google search engine to provide detailed, accurate, and trustworthy information.","understand_dependencies":null,"extract":[],"function":[{"id":"function_1","name":"GOOGLE_SEARCH","parameters":[{"name":"query","value":"What is the home country of Elon Musk?"}]}]},{"id":"step_2","thought":"Now, we need to find out the current weather in Elon Musk\'s home country.","explore_tools":"To determine the temperature of a country, I have several options that can assist me.","available_tools":["GOOGLE_SEARCH","WEATHER_INFO"],"choose_tool":"I should choose the \'WEATHER_INFO\' tool to find the temperature of Elon Musk\'s home country, as it can provide accurate and current temperature information.","understand_dependencies":"Since the \'WEATHER_INFO\' tool requires a \'location\' parameter to gather temperature details, I am dependent on the response from \'step_1.function.function_1\' to obtain the location. Therefore, I first need to extract the location from the \'step_1.function.function_1\' response.","extract":[{"id":"extract_1","context":["step_1.function.function_1"],"schema":[{"name":"country","type":"string","description":"The name of the country where Elon Musk was born or resides."}]}],"function":[{"id":"function_2","name":"WEATHER_INFO","parameters":[{"name":"location","value":"{{step_2.extract.extract_1.country}}"}]}]},{"id":"step_3","thought":"Finally, we need to format the response.","explore_tools":"To format the response, I have several options that can assist me.","available_tools":["GENERAL_LLM","FORMAT_TEXT"],"choose_tool":"I should choose the \'FORMAT_TEXT\' tool to format the response.","understand_dependencies":null,"extract":[{"id":"extract_2","context":["step_2.function.function_2"],"schema":[{"name":"temperature","type":"number","description":"The current temperature of the country."}]}],"function":[{"id":"function_3","name":"FORMAT_TEXT","parameters":[{"name":"context","value":"GOOGLE_SEARCH: <<parameters>> query: What is home country of elon musk? <<parameters>> \\\\n\\\\n Result: {{step_2.extract.extract_1.country}} \\\\n\\\\n WEATHER_INFO: <<parameters>> temperature: {{step_3.extract.extract_2.temperature}} <<parameters>>"},{"name":"query","value":"What is the current temperature in the country where Elon Musk resides?"}]}]}]}',
            },
            {
                "role": "user",
                "content": '{"query":"२०२३ के पहले छमाही में जारी किए गए सबसे लोकप्रिय स्मार्टफोन कौन से हैं?","steps":["SEARCH,NEWS: Identify the most popular smartphones released in the first half of 2023"],"functions":[{"name":"POLITICS_TECH_ANALYSIS","description":"Analyzes the intersection of politics and technology, identifying trends, sentiment, and key players.","parameters":{"type":"object","properties":[{"name":"keywords","type":"array","description":"Keywords to search for in the analysis of political and technological topics."},{"name":"date_range","type":"object","properties":[{"name":"start_date","type":"string","description":"The start date for the analysis period."},{"name":"end_date","type":"string","description":"The end date for the analysis period."}],"required":["start_date","end_date"]},{"name":"sources","type":"array","description":"List of information sources to use for the analysis, such as \'news_articles\', \'social_media\', etc."}],"required":["keywords","date_range"]}},{"name":"TECH_PRODUCT_REVIEWS","description":"Gathers reviews and ratings for technology products from various platforms.","parameters":{"type":"object","properties":[{"name":"product_name","type":"string","description":"The name of the technology product to find reviews for."},{"name":"sources","type":"array","description":"A list of sources to gather reviews from, such as \'Amazon\', \'CNET\', \'TechRadar\', etc."}],"required":["product_name"]}},{"name":"FETCH_TECH_NEWS","description":"Retrieves the latest technology news, including updates on gadgets, software, and companies in the tech industry.","parameters":{"type":"object","properties":[{"name":"topics","type":"array","description":"A list of specific tech-related topics to filter news articles."},{"name":"limit","type":"integer","description":"The maximum number of news articles to retrieve. Default is 5."}],"required":[]}},{"name":"LARGE_LANGUAGE_MODEL","description":"Interacts with a large language model to generate text, answer questions, or provide insights.","parameters":{"type":"object","properties":[{"name":"prompt","type":"string","description":"The input text or question to provide to the large language model."},{"name":"max_tokens","type":"integer","description":"The maximum number of tokens to generate in the model\'s response."},{"name":"temperature","type":"number","description":"The creativity temperature setting for the language model\'s response (typically between 0 and 1)."},{"name":"top_p","type":"number","description":"The nucleus sampling setting that controls the diversity of the response."}],"required":["prompt"]}}]}',
            },
            {
                "role": "assistant",
                "content": '',
            },
        ]
        super().__init__(
            dump_dir,
            file_prefix,
            system_prompt,
            example_messages,
            openai_func,
            max_tokens,
            verbose,
        )

    def generate(
        self,
        step_input: StepsInput | str | dict,
        dump: bool = False,
        validate: bool = True,
        strict: bool = False,
    ):
        if isinstance(step_input, (str, dict)):
            step_input = self._to_model(step_input, StepsInput)
        if not isinstance(step_input, StepsInput):
            return None
        last_user_message = try_json_dumps(step_input.model_dump(mode="json"))
        step = self.generate_response(last_user_message, dump)
        if validate:
            step = self._validate(step, StepsOutput, strict)
        return step
