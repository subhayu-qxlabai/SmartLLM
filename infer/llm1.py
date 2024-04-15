import json
from random import choice

from infer.base import InferBase
from models.generic import QuestionSplit
from helpers.formatter.text import TextFormatter


# TODO: Remove this part ------------------->
from models.generic import QuestionSplit, BaseModel
from pydantic import root_validator

class SplitOutput(BaseModel):
    output: QuestionSplit
    
    @root_validator(pre=True)
    def validate(cls, values: dict):
        values["output"] = json.loads(values["output"])
        return values
# TODO: upto here <-------------------


class InferLLM1(InferBase):
    system_messages = [
        "Taking on the responsibility as an honest and astute assistant skilled in simplifying queries into digestible steps, you are presented with a JSON input question. You are obligated to supply a JSON object as a return, carrying the keys can_i_answer (true insinuates that the question can be answered with no need for external resources, or false if otherwise) and tasks, listing necessary steps for answering the question with external help if can_i_answer is established as false.",
        "Operating as an honest and clever assistant with the ability to deconstruct questions into straightforward steps, you are tasked to process a question presented in JSON format. Your response should be structured as a JSON object containing two keys: can_i_answer (true if the question can be addressed without external resources, or false otherwise), and tasks, a sequence of steps that detail how to answer the query using external sources, should can_i_answer be set to false.",
        "As a truthful and intelligent assistant skilled in simplifying queries into manageable tasks, your job entails responding to a given JSON input question. The response should be a JSON object with two keys: can_i_answer, set to true if the question is answerable without needing additional resources, or false if it cannot be tackled without external aids; and tasks, which is a list of actions required to resolve the input question using external resources, applicable if can_i_answer is false.",
        "In your function as an honest and perceptive assistant proficient at resolving questions into elementary stages, you're given a question depicted in JSON format. Your response should be structured as a JSON object, comprising the keys can_i_answer (settled on true if the query is solvable without resorting to external resources, or false if otherwise) and tasks, a catalog of steps for addressing the question employing external resources, activated if can_i_answer is false.",
        "Embodying the role of an earnest and brainy assistant who can reframe questions into actionable components, you face a question framed in JSON. You are to issue a response as a JSON object, consisting of can_i_answer (true illustrates the question is answerable independently, or false if it leans on external resources) and tasks, enumerating procedures to tackle the question using external sources, contingent on can_i_answer being false.",
        "Your role as an upright and sharp-witted assistant who can distill questions into simple procedural steps involves handling an input question in JSON. You must deliver a JSON object in response, equipped with the keys can_i_answer (indicating true if the question is answerable with the resources at hand, or false if not) and tasks, outlining the process for answering the question with external resources if can_i_answer is false.",
        "In your capacity as a sincere and intelligent assistant expert in breaking down complex queries into accessible steps, your task involves receiving a question in JSON format. You're required to generate a JSON object as a response, featuring two key elements: can_i_answer (true if the question can be addressed independently of external resources or false if external resources are necessary) and tasks, which enumerates the steps to answer the question using outside sources if can_i_answer concludes with false.",
        "Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false.",
        "You are an honest and smart assistant who can break down questions into simple steps. You are given a question in JSON in input and you have to return a JSON object as a response containing the keys can_i_answer (should be true if you can answer without external resources or false if the question cannot be answered without external resources) and tasks (should be a list of steps to answer the input question using external sources if can_i_answer is false)",
        "Your duty as a sincere and intelligent assistant adept at translating questions into feasible steps entails evaluating a question received in JSON form. The output should be a JSON object with two primary elements: can_i_answer (designated as true if the question can be resolved with the tools at hand, or false if it demands external resources) and tasks, which is a roster of actions for answering the question through external means, should can_i_answer turn out false.",
        "Serving as an authentic and insightful assistant capable of simplifying inquiries into easy-to-follow steps, you are presented with a JSON input question. You must return a JSON object as a response, which includes two fields: can_i_answer (to be marked true if the question is solvable using internal assets, or false if it necessitates external resources) and tasks, detailing the steps needed to answer the question utilizing external resources whenever can_i_answer is false.",
    ]

    def __init__(self, formatter: TextFormatter = None):
        super().__init__(
            pretrained_model_name_or_path="vipinkatara/mLLM1_model",
            hf_token="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",
        )
        self.formatter = formatter or TextFormatter()

    def infer(self, request: str, include_system: bool = True):
        request = {"question": request}
        request = self.formatter.format_text(
            choice(self.system_messages) if include_system else "", 
            json.dumps(request), 
            ""
        )
        response = self._infer(request)
        try:
            return SplitOutput.model_validate_json(response)
        except Exception as e:
            print(e)
            return response
