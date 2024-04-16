import json
from pathlib import Path
from typing import Callable

from helpers.call_openai import call_openai_api
from dataset_gen.base.json_array_generator import JSONArrayGenerator

DEFAULT_DUMP_DIR = Path("generated/question")

class QuestionGenerator(JSONArrayGenerator):
    def __init__(
        self,
        dump_dir: str | Path = DEFAULT_DUMP_DIR,
        file_prefix: str = "q",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
    ):
        system_prompt = system_prompt or (
            "You are an excellent question generator. \n"
            # "You have to generate question that can be asked on a search engine to get latest data.  \n\n"
            "You have to generate question that you can answer yourself without any external resources. \n\n"
            "You will have to generate n random question(s) in a specified language.\n"
            "Your output should be a JSON array of strings."
        )
        example_messages = example_messages or [
            {
                "role": "user",
                "content": json.dumps(
                    {"topic": "global warming related to date", "n": 5, "language": "english"}
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
                "content": json.dumps({"topic": "Steve Jobs and Pixar", "n": 1, "language": "hindi"}),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        "कैसे स्टीव जॉब्स ने पिक्सार को एक विश्वव्यापी एनिमेशन कंपनी में बदला?",
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

    def generate(self, topic: str, n: int = 1, language: str = "english", dump: bool = False):
        """
        Generate questions based on the given topic, with an optional number of questions and dump parameter.

        Args:
            topic (str): The topic for which questions need to be generated.
            n (int, optional): The number of questions to generate, capped at 20. Defaults to 1.
            language (str, optional): The language of the questions. Defaults to "english".
            dump (bool, optional): Whether to automatically dump the generated questions. Defaults to False.

        Returns:
            list[str]: The generated questions based on the topic.
        """
        n = min(n, 20)
        last_user_message = json.dumps({"topic": topic, "n": n, "language": language})
        return self.generate_and_dump(last_user_message, 1, dump)
