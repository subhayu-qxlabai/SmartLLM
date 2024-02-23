import json
from pathlib import Path

import typer
from pydantic import BaseModel

from models.inputs import Function
from helpers.vectorstore.faisser import FaissDB, Document


app = typer.Typer()


def try_load_model(model: BaseModel, data: dict):
    try:
        return model(**data)
    except Exception as e:
        print(f"Got error {e!r} when trying to load model {model} with data {data}")


def get_function_docs(functions: list[Function]):
    docs = [
        Document(
            page_content=f"{x.name}|{x.description}",
            metadata=x.model_dump(mode="json"),
        )
        for x in functions
    ]
    return docs


def vdb_from_functions(
    functions: list[Function], dump_path: str | Path = "functions.pkl"
):
    docs = get_function_docs(functions)
    typer.echo(f"Dumping {len(docs)} functions to {dump_path.absolute().as_posix()!r}")
    return FaissDB(filename=dump_path, documents=docs)


@app.command(help="Create a vector database from a json file of functions")
def vdb_from_json(
    json_path: str = typer.Argument("functions.json", help="Path to functions json"),
    dump_path: str = typer.Option(None, help="Path to dump file. Defaults to <json_path>.pkl"),
):
    json_path: Path = Path(json_path)
    if not json_path.exists():
        raise ValueError(f"File {json_path!r} does not exist")
    functions: list[Function] = [
        try_load_model(Function, x) for x in json.load(json_path.open())
    ]
    if dump_path is None:
        dump_path = json_path.with_suffix(".pkl")
    return vdb_from_functions(functions, dump_path=dump_path)
