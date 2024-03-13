import os
from enum import Enum
from pathlib import Path
from random import shuffle

import typer
from datasets import Dataset, DatasetDict, load_dataset

from translators import DatasetTranslator
from helpers.utils import run_parallel_exec_but_return_in_order, try_json_load, get_ts_filename
from dataset_gen import DatasetGenerator, DEFAULT_TOPICS_FILE
from models.messages import ConversationFormat, messages_list_factory
from models.llm_dataset import (
    LLMType,
    LLMDataset,
    DEFAULT_DATASET_DIR,
    LLMDatasetWithTypes,
)


app = typer.Typer(no_args_is_help=True)


class ParallelismType(str, Enum):
    thread = "thread"
    process = "process"


@app.command(help="Generates dataset")
def generate(
    generate_for: int = typer.Argument(..., min=1, help="Number of topics to generate"),
    topics_file: str = typer.Option(
        "yahoo_questions_1.4M.json",
        "--topics-file",
        "-t",
        help="Path to the topics file. Must be a JSON array of strings.",
    ),
    language: str = typer.Option(
        "en", "--language", "-l", help="Language of the dataset"
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
    assert (
        isinstance(generate_for, int) and generate_for > 0
    ), "generate_for must be greater than 0"
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
        if topics_file.suffix == ".json":
            topics: list[str] = try_json_load(topics_file, [])
        if topics_file.suffix == ".txt":
            with open(topics_file, "r") as f:
                topics = [line.strip() for line in f.readlines()]
    if topics:
        shuffle(topics)
        topics = topics[:generate_for]
        typer.echo(f"Generating for {generate_for} topics")
        rows = dg.generate_parallel(topics, language, multiplier, workers, parallelism)
    else:
        generate_for = min(generate_for, 10)
        typer.echo(f"No topics found! Auto-generating for {generate_for} topics...")
        rows = dg.generate_auto(language, generate_for, multiplier, workers)
    typer.echo(f"Generated {len(rows)} rows!")


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
    add_ts: bool = typer.Option(
        False, help="Add timestamp suffix to the file name"
    ),
    quiet: bool = typer.Option(False, help="Don't print verbose messages"),
):
    dump_dir: Path = Path(dump_dir)
    suffix = (
        get_ts_filename(".json", add_random=False).name if add_ts else ".json"
    )
    dataset_map = get_dataset_map(source_dir, quiet, split_by_llm, validate_schema)
    file_prefix = f"{file_prefix}_" if file_prefix else ""
    for _type, dataset in dataset_map.items():
        filename = f"{file_prefix}{_type}{suffix}"
        existing_path = dump_dir / filename
        if merge_existing:
            if existing_path.exists() and existing_path.is_file():
                existing_dataset = LLMDataset.from_file(
                    file=existing_path.name, dir=existing_path.parent, log_errors=False
                )
                if existing_dataset:
                    typer.echo(
                        f"Merging {len(existing_dataset)} existing rows with {len(dataset)} new ones..."
                    )
                    if _type == "all" and isinstance(dataset, LLMDataset):
                        dataset = existing_dataset + dataset
                    else:
                        dataset = (
                            existing_dataset.get_llm_type_rows(LLMType(_type)) + dataset
                        )
        (
            typer.echo(
                f"Dumping {_type} dataset with {len(dataset)} rows to {filename}"
            )
            if not quiet
            else None
        )
        dataset.to_file(filename, dir=dump_dir)


@app.command(
    name="to-conv",
    help="Convert a dataset from a file or directory to conversations in a specified format",
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
        "conv", "--file-prefix", "-p", help="Prefix of the output file(s)"
    ),
    add_ts: bool = typer.Option(
        False, help="Add timestamp suffix to the file name"
    ),
    quiet: bool = typer.Option(False, help="Don't print verbose messages"),
):
    dump_dir: Path = Path(dump_dir)
    suffix = (
        get_ts_filename(".jsonl", add_random=False).name if add_ts else ".jsonl"
    )
    dataset_map = get_dataset_map(source, quiet, split_by_llm, validate_schema)
    for _type, dataset in dataset_map.items():
        conv_list = dataset.to_messages(conv_format)
        filename = f"{_type}_{conv_format.value}{suffix}"
        if file_prefix:
            filename = f"{file_prefix}_{filename}"
        if merge_existing and (dump_dir / filename).exists():
            existing_conv_list = messages_list_factory(conv_format).from_jsonl(
                dump_dir / filename
            )
            (
                typer.echo(
                    f"Merging {len(existing_conv_list)} existing conversations with {len(conv_list)} new ones for {_type}..."
                )
                if not quiet
                else None
            )
            conv_list = conv_list + existing_conv_list
        if len(conv_list) == 0:
            continue
        conv_dataset = Dataset.from_list(
            conv_list.model_dump(mode="json")["messages_list"]
        )
        (
            typer.echo(f"Dumping {len(conv_dataset)} conversations to {filename}")
            if not quiet
            else None
        )
        conv_dataset.to_json(dump_dir / filename)
        

@app.command(name="download", help="Download dataset from Hugging Face")
def download_dataset(
    dump_dir: str = typer.Argument(
        Path("dataset"), help="Directory to dump the dataset in"
    ),
    path: str = typer.Option(
        "subhayu-qxlabai/SmartLLM",
        "--path",
        "-p",
        help="Path to the dataset on Hugging Face"
    ),
    force: bool = typer.Option(
        False, 
        help="Force download even if dataset already exists"
    ),
    add_ts: bool = typer.Option(
        False, help="Add timestamp suffix to the file name"
    ),
):
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        typer.echo(f"Using HF_TOKEN from environment variable")
    if not hf_token or not hf_token.startswith("hf_"):
        hf_token: str = typer.prompt("Enter your Hugging Face token", hide_input=True, value_proc=str)
    
    dataset = load_dataset(path, token=hf_token)
    
    dump_dir: Path = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    def dump_if_missing(dst: Dataset, path: Path):
        if add_ts:
            path = path.with_name(get_ts_filename(path.name, add_random=False).name)
        if path.exists() and not force:
            typer.echo(f"Dataset already exists at {path!r}! Skipping...")
            return
        typer.echo(f"Dumping dataset to {path!r}")
        dst.to_json(path)
    
    if isinstance(dataset, DatasetDict):
        for s, d in dataset.items():
            file_name: Path = dump_dir / f"{s}.jsonl"
            dump_if_missing(d, file_name)
    elif isinstance(dataset, Dataset):
        file_name = dump_dir / f'{typer.prompt("Enter file name for the dataset", value_proc=str)}.jsonl'
        dump_if_missing(dataset, file_name)
    

@app.command(name="translate", help="Translate dataset")
def translate_dataset(
    language: str = typer.Argument(
        "hindi", help="Language to translate the dataset to"
    ),
    jsonl_file: str = typer.Argument(
        Path("dataset/llm1_alpaca.jsonl"), help="Path to the JSONL file of the dataset"
    ),
    dump_dir: str = typer.Argument(
        Path("dataset/"), help="Directory to dump the translated dataset to"
    ),
    llm_type: LLMType = typer.Option(
        None, "--llm-type", "-l", help="Dataset for which LLM type. If not specified, LLM type is inferred from the JSONL file name"
    ),
    quiet: bool = typer.Option(
        False, help="Don't print verbose messages"
    ),
    parallel_rows: int = typer.Option(
        10, "--parallel-rows", "-r", min=1, help="Number of rows to process in parallel"
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", min=1, help="Number of parallel workers to use for internal processing (inside each row)"
    ),
    page_size: int = typer.Option(
        None, "--page-size", "-p", min=1, help="Number of rows to take"
    ),
):
    if not language.lower() in DatasetTranslator.supported_languages:
        raise ValueError(
            f"Language {language} is not supported. "
            f"Supported languages are {DatasetTranslator.supported_languages}"
        )
    jsonl_file: Path = Path(jsonl_file)
    assert jsonl_file.exists(), f"{jsonl_file} does not exist"
    assert jsonl_file.is_file(), f"{jsonl_file} is not a file"
    assert jsonl_file.suffix == ".jsonl", f"{jsonl_file} is not a .jsonl file"
    dump_file = Path(dump_dir) / f"{jsonl_file.stem}_{language.lower()}.jsonl"
    dataset = LLMDataset.from_jsonl(jsonl_file, llm_type)
    dataset = dataset.get_llm_type_rows(llm_type, verbose=not quiet)
    shuffle(dataset.rows)
    if page_size is not None and page_size > 0:
        dataset = dataset.page(page_size, 0)
    typer.echo(f"Translating and dumping {len(dataset)} rows to {dump_file}")
    
    dump_file.parent.mkdir(parents=True, exist_ok=True) 
    dt = DatasetTranslator(language)
    
    def translate_and_dump(_dataset: LLMDatasetWithTypes):
        for _d in _dataset.page_iterator(1, use_tqdm=False):
            dump_file.open("a").write(
                dt
                .translate_dataset(_d, workers)
                .to_messages()
                .messages_list[0]
                .model_dump_json() + "\n"
            )
    
    typer.echo(f"Running {parallel_rows} row(s) in parallel")
    for d in dataset.page_iterator(page_size=parallel_rows):
        run_parallel_exec_but_return_in_order(
            translate_and_dump, 
            d.page_iterator(1, use_tqdm=False)
        )
