import json
from pathlib import Path
from typing import Callable

from helpers.call_openai import call_openai_api
from dataset_gen.json_array_generator import JSONArrayGenerator


class QuestionGenerator(JSONArrayGenerator):
    def __init__(
        self,
        dump_dir: str | Path = "generated/question",
        file_prefix: str = "q",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
    ):
        system_prompt = system_prompt or (
            "You are an excellent question generator. \n"
            "You have to generate question that can be asked on a search engine to get latest data. \n\n"
            "You will have to generate n random question(s).\n"
            "Your output should be a JSON array of strings."
        )
        example_messages = example_messages or [
            {
                "role": "user",
                "content": json.dumps(
                    {"topic": "global warming related to date", "n": 5}
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        "When was the term 'global warming' first coined?",
                        "What major international agreement on climate change was signed in 1997?",
                        "In what year was the Kyoto Protocol adopted, and what were its main objectives?",
                        "When did the Intergovernmental Panel on Climate Change (IPCC) release its first assessment report?",
                        "What significant event occurred in 2015 regarding global efforts to combat climate change?",
                    ]
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"topic": "Steve Jobs and Pixar", "n": 1}),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        "What was the name of the animation studio that Steve Jobs purchased and later became Pixar?",
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

    def generate(self, topic: str, n: int = 1, dump: bool = False):
        """
        Generate questions based on the given topic, with an optional number of questions and dump parameter.

        Args:
            topic (str): The topic for which questions need to be generated.
            n (int, optional): The number of questions to generate, capped at 20. Defaults to 1.
            dump (bool, optional): Whether to automatically dump the generated questions. Defaults to False.

        Returns:
            list[str]: The generated questions based on the topic.
        """
        n = min(n, 20)
        last_user_message = json.dumps({"topic": topic, "n": n})
        return self.generate_and_dump(last_user_message, 1, dump)
