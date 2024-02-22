from enum import Enum
from pathlib import Path
from random import shuffle

import typer

from helpers.utils import try_json_load
from helpers.vectorstore.faisser import FaissDB
from dataset_gen import DatasetGenerator, DEFAULT_TOPICS_FILE
from models.llm_dataset import LLMDataset, DEFAULT_DATASET_DIR, LLMType


app = typer.Typer()


class ParallelismType(str, Enum):
    thread = "thread"
    process = "process"


@app.command()
def generate(
    generate_for: int = typer.Argument(10, min=1, help="Number of topics to generate"),
    topics_file: str = typer.Option(
        "yahoo_questions_1.4M.json",
        "--topics-file",
        "-t",
        help="Path to the topics file. Must be a JSON array of strings.",
    ),
    multiplier: int = typer.Option(
        1,
        "--multiplier",
        "-m",
        min=1,
        max=20,
        help="Multiplier to apply to the generated results",
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", min=1, help="Number of workers to use"
    ),
    parallelism: ParallelismType = typer.Option(
        ParallelismType.process.value,
        "--parallelism",
        "-p",
        help="Use multiprocessing or multithreading",
    ),
    quiet: bool = typer.Option(False, help="Don't print verbose messages"),
    validate: bool = typer.Option(False, help="Validate every row in the dataset"),
    dump_rows: bool = typer.Option(True, help="Dump the rows of the dataset"),
    dump_internal: bool = typer.Option(False, help="Dump the internally generated questions, splits, steps, etc..."),
    local_embeddings: bool = typer.Option(True, help="Generate emebeddings locally"),
    dump_dir: str = typer.Option(
        DEFAULT_DATASET_DIR,
        "--dump-dir",
        "-o",
        help="Directory to dump the generated dataset in",
    ),
    generated_topics_file: str = typer.Option(
        DEFAULT_TOPICS_FILE,
        "--generated-topics-file",
        "-gt",
        help="File to store the generated topics for hash",
    ),
):
    assert generate_for > 0, "generate_for must be greater than 0"
    topics_file: Path | None = Path(topics_file) if topics_file is not None else None
    dg = DatasetGenerator(
        dump_dir=dump_dir,
        generated_topics_file=generated_topics_file,
        verbose=not quiet,
        dump_rows=dump_rows,
        dump_internal=dump_internal,
        validate=validate,
        local_embeddings=local_embeddings,
    )
    topics: list[str] = []
    if topics_file and topics_file.exists():
        topics: list[str] = try_json_load(topics_file, [])
    if topics:
        shuffle(topics)
        topics = topics[:generate_for]
        print(f"Generating for {generate_for} topics")
        rows = dg.generate_parallel(topics, multiplier, workers, parallelism)
    else:
        generate_for = min(generate_for, 10)
        typer.echo(f"No topics found! Auto-generating for {generate_for} topics...")
        rows = dg.generate_auto(generate_for, multiplier, workers)
    typer.echo(f"Generated {len(rows)} rows!")


@app.command()
def dir_to_file(
    dump_dir: str = typer.Argument(
        DEFAULT_DATASET_DIR, help="Directory to dump the files in"
    ),
    split_by_llm: bool = typer.Option(False, help="Split the dataset by LLM type"),
    source_dir: str = typer.Option(
        DEFAULT_DATASET_DIR, help="Directory to read the files from"
    ),
):
    dataset = LLMDataset.from_dir(dir=source_dir)
    if split_by_llm:
        for llm in LLMType:
            llm_dataset = dataset.get_llm(llm)
            llm_dataset.to_file(f"{llm.value}.json", dir=dump_dir)
    else:
        dataset.to_file(file="generated.json", dir=dump_dir)
