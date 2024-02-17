from multiprocessing import Pool
from pathlib import Path
from typing import Callable
from itertools import chain, zip_longest

from tqdm import tqdm

from dataset_gen.topic_generator import TopicGenerator
from dataset_gen.question_generator import QuestionGenerator
from dataset_gen.split_generator import QuestionSplitGenerator
from dataset_gen.step_input_generator import StepInputGenerator
from dataset_gen.step_output_generator import StepOutputGenerator
from models.generic import QuestionSplit
from models.llm_dataset import (
    LLMType,
    LLMDataset,
    DatasetRow,
)


class DatasetGenerator:
    def __init__(
        self,
        dump_dir: str | Path = "generated/dataset",
        generated_topics_path: str | Path = "generated/topics.txt",
        verbose=True,
        dump_rows=True,
        dump_internal=False,
        validate=True,
        local_embeddings=False,
    ):
        self.dump_dir = Path(dump_dir)
        self.generated_topics_path = Path(generated_topics_path)
        self.verbose = verbose
        self.dataset = LLMDataset()
        self.dump_rows = dump_rows
        self.dump_internal = dump_internal
        self.validate = validate
        self.local_embeddings = local_embeddings

    def _retry(self, func: Callable, *args, **kwargs):
        max_retries: int = kwargs.pop("max_retries", 3)
        result = None
        for i in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if result:
                    return result
                else:
                    if self.verbose:
                        print(f"Retrying {i+1}/{max_retries}...")
            except Exception as e:
                if "exceeds max_tokens:" in str(e):
                    return
                if self.verbose:
                    print(f"Error: {e}")

    def _generate_rows(self, topic: str, n=1):
        questions = self._retry(
            QuestionGenerator(verbose=self.verbose).generate,
            topic,
            n,
            dump=self.dump_internal,
        )
        splits = self._retry(
            QuestionSplitGenerator(verbose=self.verbose).generate,
            questions,
            dump=self.dump_internal,
            validate=self.validate,
            strict=True,
        )
        split_models: list[QuestionSplit | None] = [
            QuestionSplitGenerator()._to_model(split, QuestionSplit)
            for split in splits
        ]
        step_inputs = [
            (
                StepInputGenerator(verbose=self.verbose).generate(
                    split, dump=self.dump_internal, by_vector=self.local_embeddings
                )
                if split is not None and not split.can_i_answer
                else None
            )
            for split in split_models
        ]
        step_outputs = [
            (
                self._retry(
                    StepOutputGenerator(verbose=self.verbose, max_tokens=4090).generate,
                    step_input,
                    dump=self.dump_internal,
                    validate=self.validate,
                    strict=True,
                )
                if step_input is not None
                else None
            )
            for step_input in step_inputs
        ]

        rows: list[tuple[str | dict | None]] = list(
            zip_longest(questions, splits, step_inputs, step_outputs, fillvalue=None)
        )
        return rows

    @staticmethod
    def _map_generated(generated: list[tuple[str | dict | None]]):
        assert len(generated) == 4
        llm1_row = DatasetRow(
            llm=LLMType.LLM1, input={"question": generated[0]}, output=generated[1]
        )
        llm2_row = DatasetRow(
            uid=llm1_row.uid, llm=LLMType.LLM2, input=generated[2], output=generated[3]
        )
        return [llm1_row, llm2_row]
    
    def _load_generated_topics(self) -> list[str]:
        if not self.generated_topics_path.exists():
            return []
        with open(self.generated_topics_path, "r") as f:
            return f.readlines()
    
    def _add_generated_topics(self, topics: list[str]):
        with open(self.generated_topics_path, "a") as f:
            f.write("\n".join(topics)+"\n")

    def _add_generated_topic(self, topic: str):
        self._add_generated_topics([topic])

    def _is_new_topic(self, topic: str):
        return topic not in self._load_generated_topics()

    def generate(self, topic: str, multiplier=1) -> list[DatasetRow]:
        """
        Generates dataset rows for a given topic.

        Args:
            topic (str): The topic for which to generate dataset rows.
            multiplier (int): The number of dataset rows to generate for the topic. Default is 1.

        Returns:
            list[DatasetRow]: The list of generated dataset rows.
        """
        if not self._is_new_topic(topic):
            if self.verbose:
                print(f"Topic {topic!r} already generated")
            return []
        print(f"Generating for topic: {topic}")
        self._add_generated_topic(topic)
        generated = self._generate_rows(topic, multiplier)
        rows: list[DatasetRow] = list(
            chain(*[self._map_generated(x) for x in generated])
        )
        if self.dump_rows:
            [row.to_file(dir=self.dump_dir) for row in rows]
        else:
            self.dataset.rows.extend(rows)
        return rows
    
    def generate_parallel(self, topics: list[str], multiplier=1, n_processes=4):
        """
        Generate results in parallel using multiprocessing.Pool.

        Args:
            topics (list[str]): List of topics to generate results for.
            multiplier (int): Multiplier to apply to the generated results.
            n_processes (int): Number of parallel processes to use.

        Returns:
            list[DatasetRow]: A list of generated results.
        """
        with Pool(n_processes) as p:
            generated = p.starmap(self.generate, [(topic, multiplier) for topic in topics])
        return list(chain(*generated))
    
    def generate_auto(self, n=5, multiplier=1, n_processes=4):
        topics = TopicGenerator().generate(n, dump=True)
        if self.verbose:
            print(f"Generating for topics: {topics}")
        generated = self.generate_parallel(topics, multiplier, n_processes)
        return generated

    def dump(self):
        return self.dataset.to_dir(self.dump_dir)


if __name__ == "__main__":
    dg = DatasetGenerator(local_embeddings=True)
    try:
        dg.generate("Issac Newton", multiplier=3)
        dg.generate("French revolution", multiplier=2)
        dg.generate("Image Search", multiplier=2)
    except Exception as e:
        print(e)
    print(dg.dataset.model_dump_json())
