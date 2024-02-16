import json
from pathlib import Path
from typing import Callable

from models.generic import QuestionSplit
from helpers.utils import try_json_loads
from helpers.call_openai import call_openai_api
from dataset_gen.json_array_generator import JSONArrayGenerator


class QuestionSplitGenerator(JSONArrayGenerator):
    def __init__(
        self,
        dump_dir: str | Path = "generated/split",
        file_prefix: str = "s",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
    ):
        system_prompt = system_prompt or (
            "You are an honest and smart assistant who can tell whether or not you can answer a question without external resources.\n"
            "If you can't answer the question, you have to breakdown the questions into simple steps/tasks to get the answer.\n\n"
            "You'll be given a list of question(s) as an array of strings and you'll have to return a list of JSON objects with the following keys: question, tasks, can_i_answer"
        )
        example_messages = example_messages or [
            {
                "role": "user",
                "content": json.dumps(
                    [
                        {
                            "question": "Explain the step-by-step process of setting up a secure VPN on a Windows 10 PC."
                        },
                        {
                            "question": "Which restaurants near me have the highest ratings?"
                        },
                        {
                            "question": "Provide workout routines for effective weight loss in a month."
                        },
                        {
                            "question": "Predict the stock price of Tesla Inc. in the next month."
                        },
                        {
                            "question": "How can I troubleshoot the blue screen of death on my Dell laptop?"
                        },
                        {
                            "question": "Find the nearest coffee shop from my current location."
                        },
                        {
                            "question": "What is the average income for software engineers in San Francisco?"
                        },
                        {
                            "question": "What are the must-visit tourist attractions near my current location?"
                        },
                        {
                            "question": "What are the ratings of Inception and Interstellar?"
                        },
                    ]
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {
                            "question": "Explain the step-by-step process of setting up a secure VPN on a Windows 10 PC.",
                            "tasks": [],
                            "can_i_answer": True,
                        },
                        {
                            "question": "Which restaurants near me have the highest ratings?",
                            "tasks": [
                                "LOCATE: Identify the user's location",
                                "SEARCH: Discover the top-rated restaurants in the user's area",
                            ],
                            "can_i_answer": False,
                        },
                        {
                            "question": "Provide workout routines for effective weight loss in a month.",
                            "tasks": [],
                            "can_i_answer": True,
                        },
                        {
                            "question": "How can I troubleshoot the blue screen of death on my Dell laptop?",
                            "tasks": [],
                            "can_i_answer": True,
                        },
                        {
                            "question": "Find the nearest coffee shop from my current location.",
                            "tasks": [
                                "LOCATE: Determine the user's current location",
                                "MAPS: Locate the nearest coffee shop",
                            ],
                            "can_i_answer": False,
                        },
                        {
                            "question": "What is the average income for software engineers in San Francisco?",
                            "tasks": [],
                            "can_i_answer": True,
                        },
                        {
                            "question": "What are the must-visit tourist attractions near my current location?",
                            "tasks": [
                                "LOCATE: Identify the user's location",
                                "MAPS: Find must-visit tourist attractions near the user's location",
                            ],
                            "can_i_answer": False,
                        },
                        {
                            "question": "What are the ratings of Inception and Interstellar?",
                            "tasks": [
                                "SEARCH: Get movie ratings for Inception",
                                "SEARCH: Get movie ratings for Interstellar",
                                "COMPARE,LLM: Determine which movie has a higher rating, Inception or Interstellar",
                            ],
                            "can_i_answer": False,
                        },
                    ]
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    [
                        {
                            "question": "What are the latest advancements in renewable energy in the country that consumes the most oil?"
                        }
                    ]
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {
                            "question": "What are the latest advancements in renewable energy in the country that consumes the most oil?",
                            "tasks": [
                                "IDENTIFY: Determine the country that consumes the most oil",
                                "SEARCH: Research the latest advancements in renewable energy in that country",
                            ],
                            "can_i_answer": False,
                        }
                    ]
                ),
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
        questions: list[str],
        dump: bool = False,
        validate: bool = True,
        strict: bool = False,
    ) -> list[str | dict]:
        """
        Generate splits based on the given list of questions.

        Args:
            questions (list[str]): The list of questions to generate splits from.
            dump (bool, optional): Flag to enable dump. Defaults to False.
            validate (bool, optional): Flag to enable validation. Defaults to True.
            strict (bool, optional): Flag to enable strict validation. Defaults to False.

        Returns:
            list[str|dict]: The list of generated splits.
        """
        questions = [{"question": q.strip()} for q in questions if isinstance(q, str)]
        if not questions:
            return []
        last_user_message = json.dumps(questions)
        splits: list[dict | str] = self.generate_and_dump(last_user_message, 1, dump)
        if validate:
            splits = [
                s for s in splits if self._validate(s, QuestionSplit, strict)
            ]
        return splits
