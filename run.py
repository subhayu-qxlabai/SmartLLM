#!python3
import typer
import uvicorn
from scripts import functions, dataset


app = typer.Typer(no_args_is_help=True)

app.add_typer(functions.app, name="functions")
app.add_typer(dataset.app, name="dataset")

@app.command("api", help="Run the API server")
def run_api(
    port: int = typer.Option(8080, "--port", "-p", help="Port to run the API server on"),
    internal: bool = typer.Option(
        False, "--internal", "-i", help="Run the API server on localhost",
    ),
    reload: bool = typer.Option(
        True, "--reload", "-r", help="Reload the API server on changes",
    ),
):
    _ip = "0.0.0.0" if internal else "127.0.0.1"
    uvicorn.run("main:app", host=_ip, port=port, reload=reload)

if __name__ == "__main__":
    app()
