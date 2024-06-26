from pathlib import Path
from random import choice
from itertools import chain
from unittest.mock import MagicMock

try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = MagicMock

from helpers.vectorstore.faisser import FaissDB
from models.generic import QuestionSplit
from models.inputs import StepsInput, Function
from dataset_gen.base.model_validator import BaseModelValidator

DEFAULT_DUMP_DIR = Path("generated/step_input")
DEFAULT_VECTORESTORE_PATH = Path("functions.pkl")

class StepInputGenerator(BaseModelValidator):
    def __init__(
        self,
        dump_dir: str | Path = DEFAULT_DUMP_DIR,
        file_prefix: str = "s",
        verbose: bool = True,
        vectorstore_path: str | Path = DEFAULT_VECTORESTORE_PATH,
        vectorstore: FaissDB | None = None,
        *args,
        **kwargs,
    ):
        self.dump_path = Path(dump_dir)
        self.file_prefix = (
            file_prefix if file_prefix.endswith(".json") else f"{file_prefix}.json"
        )
        self.generated: str = None
        self.verbose = verbose
        self.vectorstore_path = Path(vectorstore_path)
        if isinstance(vectorstore, FaissDB):
            self.vdb = vectorstore
        else:
            if not self.vectorstore_path.exists():
                raise ValueError(f"vectorstore {vectorstore_path!r} does not exist")
            self.vdb = FaissDB(filename=vectorstore_path)

    @staticmethod
    def generate_embeddings(sentences: list[str] | str):
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        if isinstance(sentences, str):
            embeddings = model.encode([sentences])[0]
        elif isinstance(sentences, list) and len(sentences) > 0:
            embeddings = model.encode(sentences)
        else:
            embeddings = []
        return embeddings

    def _search_task(self, tsk: str, k: int = 3, by_vector: bool = False):
        if isinstance(SentenceTransformer, MagicMock):
            by_vector = False
        if by_vector:
            return self.vdb.similarity_search_with_score_by_vector(
                self.generate_embeddings(tsk), k=k
            )
        else:
            return self.vdb.similarity_search_with_score(tsk, k=k)

    def _generate_steps_input(
        self, split: QuestionSplit, by_vector: bool = False, *args, **kwargs
    ):
        format_func_names = ["llm", "format"]
        n_tasks = len(split.tasks)
        if 0 <= n_tasks < 2:
            k = 3
        elif 2 <= n_tasks < 4:
            k = 2
        else:
            k = 1
        function_docs = list(
            chain(
                *[
                    self._search_task(tsk, k=k, by_vector=by_vector)
                    for tsk in split.tasks + [split.question]
                ]
            )
        ) + self._search_task(format_func_names[0], k=1, by_vector=by_vector)
        unique_functions = {
            doc.page_content: Function(**doc.metadata) for doc, score in function_docs
        }
        return StepsInput(
            query=split.question,
            steps=split.tasks,
            functions=list(unique_functions.values()),
        )

    def generate(
        self,
        split: QuestionSplit | str | dict,
        by_vector: bool = False,
        *args,
        **kwargs,
    ):
        if isinstance(split, (str, dict)):
            split = self._to_model(split, QuestionSplit)
        if not isinstance(split, QuestionSplit):
            return None
        step = self._generate_steps_input(split, by_vector)
        return step.model_dump(mode="json")
