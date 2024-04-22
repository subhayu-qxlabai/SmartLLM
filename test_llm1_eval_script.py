from deepeval import evaluate
# from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from pathlib import Path
import json,time
from transformers import AutoModelForCausalLM, AutoTokenizer,BloomForCausalLM
from deepeval.models.base_model import DeepEvalBaseLLM
from helpers.text_utils import TextUtils
from deepeval.metrics import GEval
import re
from deepeval.dataset import EvaluationDataset
from datetime import datetime
from collections import defaultdict
from deepeval.metrics import BaseMetric
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from helpers.call_openai import choosed_gpt4_key
import pandas as pd

import asyncio
import logging
from typing import Optional, List, Union, Callable
from pydantic import BaseModel, Field

from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelvancyVerdict,required_params
# from deepeval.metrics.utils import trimAndLoadJson, check_test_case_params
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.telemetry import capture_metric_type
# directory_path = Path("eval_dataset_for_evaluation_metric")
    
# questions_file_path = directory_path / "questions.json"
# split_file_path = directory_path / "gold_standards.json"

# with open(questions_file_path, "r") as f:
#     questions = json.load(f)

# with open(split_file_path, "r") as f:
#     gold_standards = json.load(f)



# class SmarttLLM1(DeepEvalBaseLLM):
#     def __init__(
#         self,
#         model,
#         tokenizer
#     ):
#         self.model = model
#         self.tokenizer = tokenizer

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         '''
#         model = self.load_model()

#         device = "cuda" # the device to load the model onto

#         model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
#         model.to(device)

#         generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
#         return self.tokenizer.batch_decode(generated_ids)[0]
#         '''
#         finetuned_model = self.load_model()
#         encoded_input = self.tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
#         model_inputs = encoded_input.to('cuda')
#         generated_ids = finetuned_model.generate(
#             **model_inputs, 
#             max_new_tokens=8096, 
#             do_sample=False, 
#             pad_token_id=self.tokenizer.eos_token_id
#         )
#         decoded_output = self.tokenizer.batch_decode(generated_ids)
#         # print(f"{decoded_output[0]=}")
#         # start_string=prompt + ' {"output": "'
#         # end_string='"}' + self.tokenizer.eos_token
#         # response = TextUtils.get_middle_text(decoded_output[0], start_string, end_string)
#         # response = response.replace('\\"', '"')
#         print(decoded_output[0])
#         extracted_text = re.search(r'Text:\n(.*?)\n\n\*\*', decoded_output[0], re.DOTALL).group(1)
#         return extracted_text
        

#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def get_model_name(self):
#         return "LLM1"

# class CustomGEval(GEval):
#     async def a_measure(self, test_case):
#         actual_output = test_case.actual_output

# class AnswerRelvancyVerdict(BaseModel):
#     verdict: str
#     reason: str = Field(default=None)


class AnswerRelevancyMetric(BaseMetric):

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.using_native_model = False
            self.model = model
        else:
            self.using_native_model = True
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase) -> float:
        check_test_case_params(test_case, required_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.statements: List[str] = self._generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelvancyVerdict] = (
                    self._generate_verdicts(test_case.input)
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                capture_metric_type(self.__name__)
                return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        check_test_case_params(test_case, required_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None
        
        try:
            with metric_progress_indicator(
                self, async_mode=True, _show_indicator=_show_indicator
            ):
                self.statements: List[str] = await self._a_generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelvancyVerdict] = (
                    await self._a_generate_verdicts(test_case.input)
                )
                self.score = self._calculate_score()
                self.reason = await self._a_generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                capture_metric_type(self.__name__)
                return self.score
        except ValueError as e:
            self.score = 0
            self.reason = "Answer relevancy metric went to Json Error when generating statements."
            self.success = False
            return self.score

    async def _a_generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        return res

    def _generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        return res

    async def _a_generate_verdicts(
        self, input: str
    ) -> List[AnswerRelvancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
        else:
            res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        return verdicts

    def _generate_verdicts(self, input: str) -> List[AnswerRelvancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        return verdicts

    async def _a_generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        retry_count = 0
        while retry_count < 5:  # Retry up to 3 times
            try:
                prompt = AnswerRelevancyTemplate.generate_statements(
                    actual_output=actual_output,
                )
                if self.using_native_model:
                    res, cost = await self.model.a_generate(prompt)
                    self.evaluation_cost += cost
                else:
                    res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]
            except ValueError as e:
                retry_count += 1
                time.sleep(1)  # Wait for 1 second before retrying
                continue
        raise ValueError("Max retries over. Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.")

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["statements"]

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        # print(chat_model.invoke(prompt).content)
        res=chat_model.invoke(prompt).content
        # print(f"{res=}")
        return res

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        # print(f"{res.content=}")
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

class Bloom7b(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        finetuned_model = self.load_model()
        encoded_input = self.tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
        model_inputs = encoded_input.to('cuda')
        generated_ids = finetuned_model.generate(
            **model_inputs, 
            max_new_tokens=8096, 
            do_sample=False, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded_output = self.tokenizer.batch_decode(generated_ids)
        print(f"{decoded_output[0]=}")
        return decoded_output[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "bloom-7b"

# tokenizer = AutoTokenizer.from_pretrained("jeffreykthomas/bloom-7b-fine-tuned-stanford",device_map="auto")
# model = BloomForCausalLM.from_pretrained("jeffreykthomas/bloom-7b-fine-tuned-stanford",device_map="auto")


llm_api = choosed_gpt4_key()["api"]
# Replace these with real values
custom_model = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment=llm_api["model_id"],
    azure_endpoint=llm_api["endpoint"],
    openai_api_key=llm_api["key1"],
    # max_tokens=6000,
)

# llm1 = SmarttLLM1(model=finetuned_model, tokenizer=tokenizer)
llm1=AzureOpenAI(model=custom_model)
# llm1=Bloom7b(model=model,tokenizer=tokenizer)


# geval_coherent_metric = GEval(
#     model=llm1,
#     name="Task Coherence",
#     # criteria="Task Coherence - determine if the tasks key value given in actual output is coherent with the question given in the input.",
#     criteria="Task Coherence - determine if the tasks key value given in actual output is coherent with the question given in the input.",
#     threshold=0.5,
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
# )

geval_question_undertsanding_metric = GEval(
    model=llm1,
    name="Question Understanding with Tasks Steps Accuracy",
    # criteria="Task Coherence - determine if the tasks key value given in actual output is coherent with the question given in the input.",
    # criteria="LanguageTaskUnderstanding - determine if the steps returned in the tasks key value in the output is correct and accurate for the question given in the input. ",
    # criteria="Question Understanding with Tasks Steps Accuracy - determine if it is able to understand the input question given in any language well and has provided accurate steps to answer the question when can_i_answer is False and when can_i_answer in input is True then the tasks input value must be empty list.",
    criteria="Question Understanding with Tasks Steps Accuracy - First determine that the tasks value is empty list when can_i_answer value is True else if can_i_answer value is False then determine if the tasks value in actual output has accurate steps based on the question given in the input.",
    threshold=0.5,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

geval_format_metric = GEval(
    model=llm1,
    name="Output Format",
    # criteria="Task Coherence - determine if the tasks key value given in actual output is coherent with the question given in the input.",
    criteria="""Output Format - determine if the actual output matches the following JSON schema
    
    {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "question": {
      "type": "string"
    },
    "tasks": {
      "type": "array"
    },
    "can_i_answer": {
      "type": "boolean"
    }
  },
  "required": ["question", "tasks", "can_i_answer"]
}""",
    threshold=0.7,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

# input={"question":"\u0935\u0930\u094d\u0924\u092e\u093e\u0928 \u092e\u0947\u0902 \u092d\u093e\u0930\u0924 \u092e\u0947\u0902 \u0907\u0932\u0947\u0915\u094d\u091f\u094d\u0930\u093f\u0915 \u0935\u093e\u0939\u0928\u094b\u0902 \u092a\u0930 \u0915\u094c\u0928 \u0938\u0940 \u0938\u092c\u094d\u0938\u093f\u0921\u0940 \u0909\u092a\u0932\u092c\u094d\u0927 \u0939\u0948\u0902?"}

'''
input={"question":"\u00bfExisten incentivos para la instalaci\u00f3n de estaciones de carga de veh\u00edculos el\u00e9ctricos en Italia en 2023?"}

# input=json.dumps(input).strip()

prompt_template = f"""<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {input} [/INST] """

output={"question":"\u00bfExisten incentivos para la instalaci\u00f3n de estaciones de carga de veh\u00edculos el\u00e9ctricos en Italia en 2023?","tasks":["RESEARCH: Investigate whether there are incentives for installing electric vehicle charging stations in Italy in 2023"],"can_i_answer":False}
'''


# hallucination_metric = HallucinationMetric(threshold=0.5,model=llm1)
answerrelevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=llm1,
    include_reason=True
)


def run_evaluation(eval_dataset_path:Path):
    eval_df=pd.read_json(eval_dataset_path)
    language_type_df=eval_df[["language_type"]]

    dataset = EvaluationDataset()
    dataset.add_test_cases_from_json_file(
        file_path=eval_dataset_path,
        input_key_name="questions",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        # context_key_name="context",
        # retrieval_context_key_name="retrieval_context",
    )

    # results=evaluate(dataset, [answerrelevancy_metric,geval_metric,custom_metric])
    results=evaluate(dataset, [geval_question_undertsanding_metric,geval_format_metric])

# print(f"Evaluation Metric Results:")
# print(results)

    new_dict = defaultdict(list)

    for i in results:
        new_dict["input"].append(i.input)
        new_dict["actual_output"].append(i.actual_output)
        new_dict["context"].append(i.context)
        if i.success:
            new_dict["overall_status"].append("Paased")
        else:
            new_dict["overall_status"].append("Failed")
        for j in i.metrics:
            new_dict[f"{j.__name__} status"].append(j.is_successful())
            new_dict[f"{j.__name__} score"].append(j.score)
            new_dict[f"{j.__name__} reason"].append(j.reason)

    df = pd.DataFrame(new_dict)

    result = pd.concat([df,language_type_df],axis=1)
    current_time = datetime.now().strftime("%d%m%Y%H%M%S")

    csv_path=directory_path / f"test_LLM1_eval_test_run_{current_time}.csv"
    columns_list = ['Question Understanding with Tasks Steps Accuracy (GEval) score', 'Output Format (GEval) score']
    column_averages = result[columns_list].mean()
    print(f"{column_averages=}")
    print(type(column_averages))
    average_values={}

    average_values["Average QUTSA_GEval"]=result["Question Understanding with Tasks Steps Accuracy (GEval) score"].mean()
    average_values["Average Output_Format_GEval"]=result["Output Format (GEval) score"].mean()

    print(f"{average_values=}")

    average_df = pd.DataFrame(column_averages).transpose()

    # df_with_averages = result.append(column_averages, ignore_index=True)
    df_with_averages = pd.concat([result,average_df], ignore_index=True)
    
    df_with_averages.to_csv(csv_path,sep="|",index=False,header=True,mode="w")

    return average_values


directory_path = Path("eval_dataset_for_evaluation_metric")
eval_dataset_path = directory_path / "testing.json"

average_score=run_evaluation(eval_dataset_path)
print(average_score)