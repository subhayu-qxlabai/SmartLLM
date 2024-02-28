import json
from pathlib import Path
from typing import Callable

from helpers.call_openai import call_openai_api
from dataset_gen.base.json_array_generator import JSONArrayGenerator
from helpers.utils import clean_json_str, try_json_loads, try_json_load

DEFAULT_DUMP_DIR = Path("generated/json")

class JSONGenerator(JSONArrayGenerator):
    def __init__(
        self,
        dump_dir: str | Path = DEFAULT_DUMP_DIR,
        file_prefix: str = "x",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
    ):
        """
        Initializes the class with the specified parameters.

        Args:
            dump_dir (str | Path): The directory to dump files in.
            file_prefix (str): The prefix for the file name.
            system_prompt (str): The prompt for the system.
            example_messages (list[dict[str, str]]): A list of example messages.
            openai_func (Callable): The function to call the OpenAI API.
            max_tokens (int): The maximum number of tokens to generate.
            verbose (bool): Whether to display verbose output.

        Returns:
            None
        """
        super().__init__(
            dump_dir,
            file_prefix,
            system_prompt,
            example_messages,
            openai_func,
            max_tokens,
            verbose,
        )

    def dump(self, generated: str = ""):
        """
        Dump the generated content to a file, with an optional generated content string.

        Args:
            generated (str): The optional generated content string.

        Returns:
            str: The path to the dump file.
        """
        dump_file = self.get_dump_file()
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        generated = generated or self.generated
        if not generated:
            return
        with open(dump_file, "w") as f:
            json.dump(generated, f, indent=2)
        return dump_file

    def load(self):
        """
        Loads the list of generated items from a JSON files specified by the `dump_dir` parameter.
        """
        items: list[dict] = [
            try_json_load(x, {}) for x in self.dump_dir.rglob("*.json")
        ]
        return items

    def _generate_responses(
        self, last_user_message: str, n: int = 1, dump: bool = False
    ):
        """
        Generate text based on the input text, with optional settings for generation and dump.

        Args:
            text (str): The input text for generation.
            n (int, optional): The number of outputs to generate. Defaults to 1.
            dump (bool, optional): Whether to automatically dump the generated text. Defaults to False.

        Returns:
            list[str]: The generated text.
        """
        generated = self._get_openai_contents(last_user_message, n) or []
        generated: list[str | dict] = [
            try_json_loads(clean_json_str(x)) for x in generated
        ]
        return generated

    def generate_response(self, text: str, dump: bool = False):
        """
        Generate a new piece of text using the given input text and parameters.

        Args:
            text (str): The input text to generate from.
            dump (bool, optional): Whether to automatically dump the generated text. Defaults to False.

        Returns:
            The newly generated text.
        """
        return self.generate_and_dump(text, 1, dump)[0]
