import re
import json
import pickle
from random import randint
from functools import partial

from helpers.utils import get_nested_value, set_nested_value
from models.outputs import StepsOutput, Step, ExtractEntry
from helpers.call_openai import call_openai_api
from models.extractor import ExtractorInput
from infer import InferLLM3
import apis_scripts


class StepRunner:
    prev_cxt_regex = re.compile(r"\{\{(step_[0-9]+\.\w+\.\w+_[0-9]+.*?)\}\}")
    step_extract_pattern = re.compile(r"(step_[0-9]+)\.extract\.extract_[0-9]+")
    step_function_pattern = re.compile(r"(step_[0-9]+)\.function\.function_[0-9]+")
    check_fmt_regex = r"(.*(?:format|llm|large.*language).*)"

    def __init__(self, question: str, steps: list[Step]):
        self.question = question
        self.steps = steps
        self.context_dict = {}

    def is_format_fn(self, name):
        is_format_fn = re.match(
            self.check_fmt_regex, name, re.MULTILINE | re.IGNORECASE
        )
        return is_format_fn

    def _generate_func_output(self, name: str, **param_dict):
        if self.is_format_fn(name):
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant who specializes in generating human readable answers.\nGiven a question and context, you need to use the context to formulate the answer.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": self.question, "context": param_dict}
                    ),
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant who is exceptional executing tasks. You will be given a function name with some parameters, and you have to generate some satisfactory output for the function.\nYou just generate output, no need to give the id or name of the function in your response.\nYou can respond in JSON or any format you like.",
                },
                {
                    "role": "user",
                    "content": '{"function":"ARCHITECTURE_NEWS_RETRIEVAL","input":{"topics":["tallest building","construction"]}}',
                },
                {
                    "role": "assistant",
                    "content": '[{"topic":"tallest building","news":[{"title":"New Record: The World\'s Tallest Building Now Complete","location":"Dubai, United Arab Emirates","building_name":"Burj Khalifa 2","height":"838 meters","completion_date":"2022-11-14","summary":"Dubai once again shatters records with the completion of Burj Khalifa 2, standing at an astonishing 838 meters. This architectural marvel has taken the title of the world\'s tallest building, surpassing the original Burj Khalifa."}]},{"topic":"construction","news":[{"title":"Revolutionary Eco-Friendly Construction Material Hits the Market","location":"Global","release_date":"2023-04-10","summary":"A groundbreaking eco-friendly construction material, designed to reduce carbon footprint significantly, is now available. This innovation is expected to revolutionize the construction industry, making sustainable building practices more accessible worldwide."},{"title":"Construction of the Underwater Train Tunnel Begins","location":"Between Finland and Estonia","start_date":"2023-02-01","expected_completion":"2030","summary":"In an ambitious effort to enhance connectivity in Europe, construction has started on an underwater train tunnel between Finland and Estonia. This engineering feat, expected to complete by 2030, will drastically shorten travel times and foster closer ties between the two countries."}]}]',
                },
                {
                    "role": "user",
                    "content": '{"function":"AWARDS_INFORMATION_RETRIEVAL","input":{"award_name":"Fif World Cup","year":2022,"category":"National"}}',
                },
                {
                    "role": "assistant",
                    "content": '{"award_name":"FIFA World Cup","year":2022,"category":"National","winner":{"country":"Argentina","coach":"Lionel Scaloni","top_scorer":{"name":"Lionel Messi","goals":7},"final_match":{"opponent":"France","score":"3-3 (4-2 on penalties)","date":"2022-12-18","location":"Lusail Iconic Stadium, Qatar","summary":"In a thrilling final that ended in a draw during regular time, Argentina won against France on penalties. Lionel Messi played a pivotal role throughout the tournament and in the final, contributing significantly to Argentina\'s victory."}}}',
                },
                {
                    "role": "user",
                    "content": '{"function":"BUSINESS_ARTICLES_RETRIEVAL","input":{"topic":"richest people","date_range":{"start_date":"2023-01-01","end_date":"2023-12-31"},"source":"Forbes"}}',
                },
                {
                    "role": "assistant",
                    "content": '[{"topic":"richest people","articles":[{"title":"Forbes 2023 Billionaires List: The Richest People in the World","date":"2023-04-05","source":"Forbes","summary":"Forbes has released its 2023 Billionaires List, showcasing the wealthiest individuals globally. This year, the list is topped by Elon Musk, with an estimated net worth of $219 billion, closely followed by Jeff Bezos and Bernard Arnault. The report highlights the dynamic shifts in wealth, driven by the tech industry\'s growth and global economic changes.","key_figures":["Elon Musk - $219 billion","Jeff Bezos - $171 billion","Bernard Arnault - $158 billion"]},{"title":"Tech Titans Lead Forbes\' Richest in 2023","date":"2023-03-10","source":"Forbes","summary":"The 2023 Forbes Billionaires List is dominated by leaders in the technology sector, underscoring the immense wealth generated by this industry. Elon Musk, Jeff Bezos, and Mark Zuckerberg feature prominently, reflecting their companies\' success and the broader tech industry\'s impact on global wealth.","key_figures":["Elon Musk","Jeff Bezos","Mark Zuckerberg"]}]}]',
                },
                {
                    "role": "user",
                    "content": '{"input":{"region":"New York","time_frame":"current"},"function":"GENERATE_CLIMATE_REPORT"}',
                },
                {
                    "role": "assistant",
                    "content": '{"region":"New York","time_frame":"current","climate_report":{"temperature":"15°C","humidity":"68%","precipitation":"0mm","wind_speed":"10 km/h","forecast":"Cloudy skies with a chance of rain in the evening. Mild temperatures with moderate humidity levels. Light to moderate winds expected throughout the day."}}',
                },
                {
                    "role": "user",
                    "content": json.dumps({"function": name, "input": param_dict}),
                },
            ]

        response = call_openai_api(messages)
        if response is None:
            return
        response = response.choices[0].message.content
        try:
            return json.loads(response)
        except:
            return response

    def call_function(self, name: str, param_dict: dict):
        for key, value in param_dict.items():
            if not isinstance(value, str):
                continue
            prev_ctxs: list[str] = self.prev_cxt_regex.findall(value)
            if not prev_ctxs:
                continue
            for cxt in prev_ctxs:
                cxt_org = cxt
                if "function.function" in cxt and not cxt.endswith(".output"):
                    cxt += ".output"
                value = value.replace(
                    f"{{{{{cxt_org}}}}}", str(get_nested_value(self.context_dict, cxt))
                )
            param_dict.update({key: value})
        # TODO: Finish this function execution logic
        # func=getattr(load_step_outputs,"create_function_dict_for_every_steps") #this will be used if the function is defined in the different module
        # func=globals()[name] #this will be used if the function is defined in the same module
        # res=func(**dict) # run the function
        # res = self._generate_func_output(
        #     name, **param_dict
        # )  # TODO: remove this afterwards

        # func=globals()[name](**params_dict)
        default_func=partial(self._generate_func_output,name)
        func=getattr(apis_scripts,name,default_func)(**param_dict)
        print(name)
        print(param_dict)
        # func=getattr(apis_scripts,name,default_func)(**param_dict)

        # print(f"{func=}")
        
        return func

    def extract_data(self, _e: ExtractEntry):
        e = _e.model_dump(mode="json")
        e["context"] = [
            get_nested_value(self.context_dict, f"{x}.output" if "function" in x else x)
            for x in _e.context
        ]
        # Call 3rd LLM to extract data from context and return the filled schema
        extracted_data = self.llm3.infer(ExtractorInput(**e))
        # # extracted_data = {x.name: f"'{x.name}-VALUE'" for x in _e.schema}
        return extracted_data

    def run_extracts(self, step: Step):
        for _e in step.extract:
            set_nested_value(
                self.context_dict, f"{step.id}.extract.{_e.id}", self.extract_data(_e)
            )

    def run_functions(self, step: Step):
        for func in step.function:
            params_dict = {}
            for _p in func.parameters:
                params_dict.update({_p.name: _p.value})
            res = self.call_function(func.name, params_dict)
            set_nested_value(
                self.context_dict,
                f"{step.id}.function.{func.id}",
                {"input": params_dict, "function_name": func.name, "output": res},
            )

    def run_steps(self):
        # print(f"{self.steps=}")
        self.llm3 = InferLLM3()
        for step in self.steps:
            step_dump = step.model_dump(mode="json")
            funcs_cxts = self.step_extract_pattern.findall(str(step_dump["function"]))
            extrs_cxts = self.step_function_pattern.findall(str(step_dump["extract"]))

            extracts_used_in_funcs = step.id in funcs_cxts
            funcs_used_in_extracts = step.id in extrs_cxts

            if extracts_used_in_funcs and not funcs_used_in_extracts:
                self.run_extracts(step)
                self.run_functions(step)
            elif funcs_used_in_extracts and not extracts_used_in_funcs:
                self.run_functions(step)
                self.run_extracts(step)
            else:
                self.run_extracts(step)
                self.run_functions(step)
        del self.llm3
        return self.context_dict


if __name__ == "__main__":
    step_outputs: list[StepsOutput] = pickle.load(open("step_outputs.pkl", "rb"))

    i = randint(0, len(step_outputs) - 1)
    # i = 463
    print(step_outputs[i].model_dump_json(indent=4), "\n\n")
    runner = StepRunner(step_outputs[i].steps)
    runner.run_steps()
    res = json.dumps(runner.context_dict, indent=4)
    print(i, res)
