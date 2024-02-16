import json
from pathlib import Path
from random import choices
from typing import Callable

from helpers.call_openai import call_openai_api
from dataset_gen.json_array_generator import JSONArrayGenerator


class TopicGenerator(JSONArrayGenerator):
    domains = [
        "geography",
        "history",
        "scientific principles",
        "literature",
        "arts",
        "philosophy",
        "psychology",
        "sociology",
        "economics",
        "politics",
        "languages",
        "cultural studies",
        "technological advancements",
        "market trends",
        "current events",
        "medicine",
        "engineering",
        "computer science",
        "agriculture",
        "environmental science",
        "architecture",
        "law and justice",
        "business management",
        "cultural nuances",
        "situational awareness",
        "problem-solving skills",
        "communication skills",
        "analytical techniques",
        "critical thinking",
        "creativity",
        "hypothesis evaluation",
        "machine learning",
        "data analysis",
        "statistical modeling",
        "ethical reasoning applied across various contexts",
        "professional ethics",
        "societal impact",
        "moral dilemmas",
    ]
    def __init__(
        self,
        dump_dir: str | Path = "generated/topic",
        file_prefix: str = "t",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
        domains: list[str] = [],
        n_domains: int = 5,
    ):
        self.domains = domains or self.domains
        self.chosen_domains = choices(self.domains, k=n_domains)
        system_prompt = system_prompt or (
            "You are an excellent topics generator.\n"
            "You have to generate question that can be asked on a search engine to get latest data.\n"
            "You will be given a number `n` have to generate n random topics(s).\n"
            "Your output should be a JSON array of strings.\n"
            "You must always generate new topics that are not previously seen.\n"
            "Each topic should be a sentence (2-3 words) and not a question. The topics can lean more on current events.\n"
            f"Topics can span over various fields like {', '.join(self.chosen_domains)} and more."
        )
        example_messages = example_messages or [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": '["Ancient Egypt"]'},
            {"role": "user", "content": "4"},
            {
                "role": "assistant",
                "content": '["Space Race", "Quantum Computing", "Albert Einstein", "Image Search"]',
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

    def generate(self, n: int = 5, dump: bool = False):
        """
        Generate random topics.

        Args:
            n (int, optional): The number of random topics to generate. Defaults to 5.
            dump (bool, optional): Whether to dump the generated topics to a file. Defaults to False.

        Returns:
            list[str]: A list of randomly generated topics.

        Note:
            The function generates new topics based on a system prompt and stores them in a JSON array.
            The number of topics generated is capped at 20.
        """
        n = min(n, 20)
        last_user_message = json.dumps(n)
        return self.generate_and_dump(last_user_message, 1, dump)
