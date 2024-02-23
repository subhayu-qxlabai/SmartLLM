from enum import Enum
from pathlib import Path
from random import shuffle

import typer
from datasets import Dataset

from helpers.utils import try_json_load, get_ts_filename
from dataset_gen import DatasetGenerator, DEFAULT_TOPICS_FILE
from models.llm_dataset import (
    LLMDataset,
    DEFAULT_DATASET_DIR,
    LLMDatasetWithTypes,
    LLMType,
)


app = typer.Typer()


class ParallelismType(str, Enum):
    thread = "thread"
    process = "process"


class ConversationFormat(str, Enum):
    alpaca = "alpaca"
    openai = "openai"


@app.command(help="Generates dataset")
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
    dump_internal: bool = typer.Option(
        False, help="Dump the internally generated questions, splits, steps, etc..."
    ),
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


def _merge_existing_dataset(current_dataset: LLMDataset, existing_path: Path):
    if existing_path.exists() and existing_path.is_file():
        existing_dataset = LLMDataset.from_file(
            file=existing_path.name, dir=existing_path.parent, log_errors=False
        )
        if existing_dataset:
            current_dataset = existing_dataset + current_dataset
    return current_dataset


def get_dataset_map(
    source: str | Path, quiet=False, split_by_llm=True, validate_schema=True
):
    source: Path = Path(source)
    if not source.exists():
        raise ValueError(f"{source} does not exist!")

    if source.is_dir():
        dataset = LLMDataset.from_dir(dir=source, log_errors=not quiet)
    elif source.is_file():
        dataset = LLMDataset.from_file(
            file=source.name, dir=source.parent, log_errors=not quiet
        )
    else:
        raise ValueError(f"{source} is not a directory or file!")

    if not dataset:
        raise ValueError(f"Failed to load dataset from {source}")

    dataset_map: dict[str, LLMDataset | LLMDatasetWithTypes] = {}

    if split_by_llm:
        for llm in LLMType:
            if validate_schema:
                dataset_map[llm.value] = dataset.get_llm_type_rows(llm)
            else:
                dataset_map[llm.value] = dataset.get_llm(llm)
    else:
        dataset_map["all"] = dataset
    return dataset_map


@app.command(help="Convert a dataset from a directory to a file")
def to_file(
    source_dir: str = typer.Argument(
        DEFAULT_DATASET_DIR, help="Directory to read the files from"
    ),
    dump_dir: str = typer.Argument(
        DEFAULT_DATASET_DIR, help="Directory to dump the files in"
    ),
    split_by_llm: bool = typer.Option(False, help="Split the dataset by LLM type"),
    validate_schema: bool = typer.Option(
        True, help="Validate the schema before dumping"
    ),
    merge_existing: bool = typer.Option(
        True, help="Merge the new dataset with the existing one if it exists"
    ),
    file_prefix: str = typer.Option(
        "dataset", "--file-prefix", "-p", help="Prefix of the output file(s)"
    ),
    add_timestamp: bool = typer.Option(
        False, help="Add timestamp suffix to the file name"
    ),
    quiet: bool = typer.Option(False, help="Don't print verbose messages"),
):
    dump_dir: Path = Path(dump_dir)
    suffix = (
        get_ts_filename(".json", add_random=False).name if add_timestamp else ".json"
    )
    dataset_map = get_dataset_map(source_dir, quiet, split_by_llm, validate_schema)
    for _type, dataset in dataset_map.items():
        filename = f"{file_prefix}_{_type}{suffix}"
        if merge_existing:
            dataset = _merge_existing_dataset(dataset, dump_dir / filename)
        dataset.to_file(filename, dir=dump_dir)


@app.command(
    help="Convert a dataset from a file or directory to conversations in a specified format"
)
def to_conversations(
    source: str = typer.Argument(
        DEFAULT_DATASET_DIR, help="Directory or file to read the dataset from"
    ),
    dump_dir: str = typer.Argument(
        Path("dataset"), help="Directory to dump the generated dataset in"
    ),
    conv_format: ConversationFormat = typer.Option(
        ConversationFormat.alpaca.value,
        "--format",
        "-f",
        help="Format to convert the dataset to",
    ),
    split_by_llm: bool = typer.Option(True, help="Split the dataset by LLM type"),
    validate_schema: bool = typer.Option(
        True, help="Validate the schema before dumping"
    ),
    merge_existing: bool = typer.Option(
        True, help="Merge the new dataset with the existing one if it exists"
    ),
    file_prefix: str = typer.Option(
        None, "--file-prefix", "-p", help="Prefix of the output file(s)"
    ),
    add_timestamp: bool = typer.Option(
        False, help="Add timestamp suffix to the file name"
    ),
    quiet: bool = typer.Option(False, help="Don't print verbose messages"),
):
    dump_dir: Path = Path(dump_dir)
    suffix = (
        get_ts_filename(".jsonl", add_random=False).name if add_timestamp else ".jsonl"
    )
    dataset_map = get_dataset_map(source, quiet, split_by_llm, validate_schema)
    for _type, dataset in dataset_map.items():
        if conv_format == ConversationFormat.alpaca:
            conv_dataset = Dataset.from_list(
                [x.model_dump(mode="json") for x in dataset.to_alpaca()]
            )
        elif conv_format == ConversationFormat.openai:
            conv_dataset = Dataset.from_list(
                dataset.to_messages().model_dump(mode="json")["messages_list"]
            )
        else:
            raise ValueError(f"Invalid conversation format: {conv_format}")
        filename = f"{_type}_{conv_format.value}{suffix}"
        if file_prefix is not None:
            filename = f"{file_prefix}_{filename}"
        conv_dataset.to_json(dump_dir / filename)
