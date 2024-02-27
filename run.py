#!python3
import typer
from scripts import functions, dataset


app = typer.Typer(no_args_is_help=True)

app.add_typer(functions.app, name="functions")
app.add_typer(dataset.app, name="dataset")


if __name__ == "__main__":
    app()
