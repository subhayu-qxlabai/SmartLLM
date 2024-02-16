import re
import json
import pickle
from typing import Any
from random import randint

from utils import get_nested_value, set_nested_value
from models.outputs import StepsOutput, Step, ExtractEntry
from models.extractor import ExtractorInput
from infer.extractor import  extract_contexts


class StepRunner:
    prev_cxt_regex = re.compile(r"\{\{(step_[0-9]+\.\w+\.\w+_[0-9]+.*?)\}\}")
    
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.context_dict = {}

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
                    f"{{{{{cxt_org}}}}}", 
                    str(get_nested_value(self.context_dict, cxt))
                )
            param_dict.update({key: value})
        # TODO: Finish this function execution logic
        # func=getattr(load_step_outputs,"create_function_dict_for_every_steps") #this will be used if the function is defined in the different module
        # func=globals()[name] #this will be used if the function is defined in the same module
        # res=func(**dict) # run the function
        res = f"'{name}-OUTPUT'" # fake output
        return res

    def extract_data(self, _e: ExtractEntry):
        e = _e.model_dump(mode="json")
        e["context"] = [get_nested_value(self.context_dict, f"{x}.output") for x in _e.context]
        # Call 3rd LLM to extract data from context and return the filled schema
        extracted_data: dict[str, Any] = extract_contexts(ExtractorInput(**e))
        # # extracted_data = {x.name: f"'{x.name}-VALUE'" for x in _e.schema}
        return extracted_data

    def run_steps(self):
        for step in self.steps:
            step_id = step.id
            for func in step.function:
                params_dict = {}
                for _p in func.parameters:
                    params_dict.update({_p.name: _p.value})
                res = self.call_function(func.name, params_dict)
                set_nested_value(
                    self.context_dict,
                    f"{step_id}.function.{func.id}",
                    {"input": params_dict, "function_name": func.name, "output": res},
                )
            for _e in step.extract:
                set_nested_value(
                    self.context_dict, f"{step_id}.extract.{_e.id}", self.extract_data(_e)
                )
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
