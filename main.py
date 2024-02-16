from itertools import chain

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query

from vectorstore.faisser import FaissDB
from models.generic import QuestionSplit
from models.inputs import StepsInput, Function
from models.outputs import StepsOutput
from step_runner import StepRunner
from infer.steps_generator import get_steps
from infer.question_breaker import break_question
from helpers.middleware import ProcessTimeMiddleware

app = FastAPI()
app.add_middleware(ProcessTimeMiddleware)

vdb = FaissDB(filename="vectorstore.pkl")


class OutModel(BaseModel):
    steps_input: StepsInput | str | None = None
    steps_output: StepsOutput | str | None = None
    context_dict: dict | None = None


@app.post("/process_question")
async def process_question(question: str = Query(..., title="User Question")):
    split: QuestionSplit = break_question(question)

    if not isinstance(split, QuestionSplit):
        print(split)
        raise HTTPException(status_code=400, detail="Failed to split the question")

    function_docs = list(chain(*[vdb.similarity_search(task, k=3) for task in split.tasks]))
    functions = set([Function.model_validate(doc.metadata) for doc in function_docs])
    input_schema = StepsInput(query=split.question, steps=split.tasks, functions=functions)
    
    step_output: StepsOutput = get_steps(input_schema)

    if not isinstance(step_output, StepsOutput):
        print(step_output)
        raise HTTPException(status_code=400, detail="Failed to generate steps")

    runner = StepRunner(step_output.steps)
    runner.run_steps()

    response = OutModel(steps_input=input_schema, steps_output=step_output, context_dict=runner.context_dict)
    filepath = f"run_logs/{question.replace(' ', '_').lower()}.json"
    with open(f"run_logs/{input_schema.query}", "w") as f:
        f.write(response.model_dump_json(indent=4))
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
