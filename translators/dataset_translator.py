from pathlib import Path
import re
from tqdm import tqdm

from helpers.utils import run_parallel_exec_but_return_in_order
from translators.translator import Translator
from models.llm_dataset import (
    LLMDatasetWithTypes,
    LLM1DatasetRow,
    LLM2DatasetRow,
    LLM3DatasetRow,
    LLMType,
    Question,
    QuestionSplit,
    StepsInput,
    StepsOutput,
    ExtractorInput,
)


class DatasetTranslator(Translator):
    """Class for translating datasets and rows"""
    strings_to_reject = ["markdown", "html", "text"]
    strings_to_reject = re.compile(
        f"\s*{'|'.join(strings_to_reject)}\s*", flags=re.IGNORECASE
    )
    language_to_replace = re.compile(
        f"\s*{'|'.join(Translator.supported_languages)}\s*", flags=re.IGNORECASE
    )
    
    model_skipkeys_map = {
        Question: [],
        QuestionSplit: ["tasks"],
        StepsInput: ["steps", "functions"],
        StepsOutput: [
            "id",
            "name",
            "overview",
            "thought",
            "explore_tools",
            "available_tools",
            "choose_tool",
            "understand_dependencies",
            "extract",
        ],
        ExtractorInput: ["schema"],
    }

    def __init__(self, language: str) -> None:
        """
        Initializes the class with the translation language.

        Parameters:
            language (str): The language to translate to.

        Returns:
            None
        """
        super().__init__(language, self._predicate)

    def _predicate(self, text: str):
        # function_or_extract_references = re.findall(r"{{[\w\.]+}}", text)
        # if function_or_extract_references:
        #     return text
        if len(text) <= 2:
            return text
        if self.strings_to_reject.search(text):
            return text
        if self.language_to_replace.search(text):
            return self.language_to_replace.sub(self.language, text)

    def translate(
        self,
        data: (
            Question | QuestionSplit | StepsInput | StepsOutput | ExtractorInput | dict
        ),
        workers: int = 4,
    ) -> Question | QuestionSplit | StepsInput | StepsOutput | ExtractorInput | dict:
        """
        Translate the given data using multiple workers if specified. 

        Args:
            data: The input data to be translated, which can be of type Question, QuestionSplit, StepsInput, StepsOutput, ExtractorInput, or dict.
            workers: The number of worker processes to use for translation. Defaults to 4.

        Returns:
            The translated data, which can be of type Question, QuestionSplit, StepsInput, StepsOutput, ExtractorInput, or dict.
        """
        return self._translate_any(
            data, self.model_skipkeys_map.get(type(data), []), workers
        )

    def translate_many(
        self,
        data_list: list[
            Question | QuestionSplit | StepsInput | StepsOutput | ExtractorInput | dict
        ],
        workers: int = 4,
    ) -> list[
        Question | QuestionSplit | StepsInput | StepsOutput | ExtractorInput | dict
    ]:
        """
        A function to translate a list of data with multiple options for input and return types.
        
        Args:
            data_list: A list of Question, QuestionSplit, StepsInput, StepsOutput, ExtractorInput, or dict.
            workers: An integer representing the number of workers for parallel execution.
        
        Returns:
            The same list of data with the nested strings translated.
        """
        internal_workers = max(abs(5 - min(5, workers)), 1)
        return run_parallel_exec_but_return_in_order(
            self.translate, data_list, internal_workers, max_workers=workers
        )
    def translate_row(self, row: LLM1DatasetRow | LLM2DatasetRow | LLM3DatasetRow, workers: int = 4):
        """
        Translate a given dataset row and return the translated row.

        Args:
            row (LLM1DatasetRow | LLM2DatasetRow | LLM3DatasetRow): The dataset row to translate.
            workers (int): The number of workers to use for translation. Default is 4.

        Returns:
            LLM1DatasetRow | LLM2DatasetRow | LLM3DatasetRow: The translated dataset row.
        """
        vals_to_translate = []
        if type(row) == LLM1DatasetRow:
            vals_to_translate.extend([row.output])
        if type(row) in [LLM2DatasetRow, LLM3DatasetRow]:
            vals_to_translate.extend([row.input, row.output])
        if not vals_to_translate:
            return row
        vals_to_translate = [
            x for x in vals_to_translate
            if type(x)
            in [Question, QuestionSplit, StepsInput, StepsOutput, ExtractorInput, dict]
        ]
        translated = self.translate_many(vals_to_translate, workers=workers)
        if not translated:
            return row
        if (
            type(translated[0]) != type(vals_to_translate[0]) 
            and type(translated[-1]) != type(vals_to_translate[-1])
        ):
            translated = translated[::-1]
        if type(row) == LLM1DatasetRow:
            row.input = Question(**translated[0].model_dump())
            row.output = translated[0]
        if type(row) in [LLM2DatasetRow, LLM3DatasetRow]:
            row.input, row.output = translated
        return row

    def translate_dataset(self, dataset: LLMDatasetWithTypes, workers: int = 4):
        """
        Translates the given LLMDatasetWithTypes using the specified number of workers.

        Args:
            dataset (LLMDatasetWithTypes): The dataset to be translated.
            workers (int, optional): The number of workers to use for translation. Defaults to 4.

        Returns:
            LLMDatasetWithTypes: The translated dataset.
        """
        return LLMDatasetWithTypes(
            rows=[self.translate_row(row, workers=workers) for row in dataset.rows]
        )
    