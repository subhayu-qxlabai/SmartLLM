from pathlib import Path
from multiprocessing import Pool
from typing import Callable, Literal
from itertools import chain, zip_longest

from dataset_gen.smart_llm.topic_generator import TopicGenerator
from dataset_gen.smart_llm.question_generator import QuestionGenerator
from dataset_gen.smart_llm.split_generator import QuestionSplitGenerator
from dataset_gen.smart_llm.step_input_generator import StepInputGenerator
from dataset_gen.smart_llm.step_output_generator import StepOutputGenerator
from helpers.utils import run_parallel_exec_but_return_in_order
from helpers.vectorstore.faisser import FaissDB
from models.generic import QuestionSplit
from models.llm_dataset import (
    LLMType,
    LLMDataset,
    DatasetRow,
    DEFAULT_DATASET_DIR,
)

DEFAULT_TOPICS_FILE = "topics.txt"

class DatasetGenerator:
    def __init__(
        self,
        dump_dir: str | Path = DEFAULT_DATASET_DIR,
        generated_topics_file: str | Path = DEFAULT_TOPICS_FILE,
        verbose=True,
        dump_rows=True,
        dump_internal=False,
        validate=True,
        local_embeddings=False,
        vectorstore: FaissDB | None = None,
    ):
        """
        Initialize the class with the given parameters.

        Parameters:
            dump_dir (str | Path): The directory to dump the generated dataset.
            generated_topics_file (str | Path): The file to store the generated topics. Used for hash.
            verbose (bool): Whether to print verbose messages.
            dump_rows (bool): Whether to dump the rows of the dataset.
            dump_internal (bool): Whether to dump the internal representation of the dataset.
            validate (bool): Whether to validate the dataset.
            local_embeddings (bool): Whether to use local embeddings.
            vectorstore (FaissDB | None): The vector store database, or None if not provided.
        """
        self.dump_dir = Path(dump_dir)
        self.generated_topics_path = self.dump_dir / generated_topics_file
        if not self.generated_topics_path.exists():
            self.generated_topics_path.parent.mkdir(parents=True, exist_ok=True)
            self.generated_topics_path.touch(exist_ok=True)
        self.verbose = verbose
        self.dataset = LLMDataset()
        self.dump_rows = dump_rows
        self.dump_internal = dump_internal
        self.validate = validate
        self.local_embeddings = local_embeddings
        self.vectorstore = vectorstore

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

    def _generate_rows(self, topic: str, n=1, language: str = "english"):
        questions = self._retry(
            QuestionGenerator(verbose=self.verbose).generate,
            topic,
            n=n,
            language=language,
            dump=self.dump_internal,
        )
        splits = self._retry(
            QuestionSplitGenerator(verbose=self.verbose).generate,
            questions,
            dump=self.dump_internal,
            validate=self.validate,
            strict=True,
        )
        if not splits:
            return []
        split_models: list[QuestionSplit | None] = [
            QuestionSplitGenerator()._to_model(split, QuestionSplit) for split in splits
        ]
        step_inputs = [
            (
                StepInputGenerator(
                    verbose=self.verbose, vectorstore=self.vectorstore
                ).generate(
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
            return set(f.read().splitlines())

    def _add_generated_topics(self, topics: list[str]):
        with open(self.generated_topics_path, "a") as f:
            f.write("\n".join(topics) + "\n")

    def _add_generated_topic(self, topic: str):
        self._add_generated_topics([topic])

    def _is_new_topic(self, topic: str):
        existing_topics = self._load_generated_topics()
        return topic not in existing_topics

    def generate(self, topic: str, language: str = "english", multiplier=1) -> list[DatasetRow]:
        """
        Generates dataset rows for a given topic.

        Args:
            topic (str): The topic for which to generate dataset rows.
            language (str): The language of the dataset rows.
            multiplier (int): The number of dataset rows to generate for the topic. Default is 1.

        Returns:
            list[DatasetRow]: The list of generated dataset rows.
        """
        if not self._is_new_topic(topic):
            if self.verbose:
                print(f"Topic {topic!r} already generated or in progress or failed.")
            return []
        print(f"Generating for topic: {topic}")
        self._add_generated_topic(topic)
        generated = self._generate_rows(topic, multiplier, language)
        rows: list[DatasetRow] = list(
            chain(*[self._map_generated(x) for x in generated])
        )
        if self.dump_rows:
            [row.to_file(dir=self.dump_dir) for row in rows]
        else:
            self.dataset.rows.extend(rows)
        return rows

    def generate_parallel(
        self,
        topics: list[str],
        language: str = "english",
        multiplier=1,
        workers=4,
        parallelism: Literal["thread", "process"] = "thread",
    ):
        """
        Generate results in parallel using multiprocessing.Pool.

        Args:
            topics (list[str]): List of topics to generate results for.
            language (str): Language of the generated results.
            multiplier (int): Multiplier to apply to the generated results.
            workers (int): Number of parallel workers to use.
            parallelism (Literal["thread", "process"], optional): Use multiprocessing or multithreading.

        Returns:
            list[DatasetRow]: A list of generated results.
        """
        if parallelism == "thread":
            generated: list[list[DatasetRow]] = run_parallel_exec_but_return_in_order(
                self.generate, topics, language, multiplier, max_workers=workers, quiet=not self.verbose,
            )
        elif parallelism == "process":
            with Pool(workers) as p:
                generated = p.starmap(
                    self.generate, [(topic, language, multiplier) for topic in topics]
                )
        else:
            raise ValueError("parallelism must be 'thread' or 'process'")
        return list(chain(*generated))

    def generate_auto(self, language: str = "english", n=5, multiplier=1, workers=4):
        """
        Automatically generates dataset.
        
        Args:
            language (str): Language of the generated results.
            n (int): Number of topics to generate results for.
            multiplier (int): Multiplier to apply to the generated results.
            workers (int): Number of parallel workers to use.
        Returns:
            list[DatasetRow]: A list of generated results.
        """
        topics = TopicGenerator().generate(n, dump=True)
        if self.verbose:
            print(f"Generating for topics: {topics}")
        generated = self.generate_parallel(topics, language, multiplier, workers)
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
