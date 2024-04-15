import re
import json
from itertools import chain
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException, Query

from helpers.vectorstore.faisser import FaissDB
from models.generic import QuestionSplit
from models.inputs import StepsInput, Function
from models.outputs import StepsOutput
from step_runner import StepRunner
from infer.generic import ask_llm
from infer import InferLLM1, InferLLM2
from helpers.middleware import ProcessTimeMiddleware
from helpers.utils import get_ts_filename, remove_special_chars
from pathlib import Path


app = FastAPI()
app.add_middleware(ProcessTimeMiddleware)

vdb = FaissDB(filename="functions2.pkl")

unanswered_regex = re.compile(
    r"(as\s*of\s*my\s*last|as\s*of\s*my\s*current|last\s*knowledge|knowledge.*\d{4})", 
    flags=re.IGNORECASE | re.DOTALL,
)

class QA(BaseModel):
    question: str
    answer: str

class OutModel(BaseModel):
    split: QuestionSplit | None = None
    steps_input: StepsInput | str | None = None
    steps_output: StepsOutput | str | None = None
    context_dict: dict | None = None
    response: str | None = None
    

@app.get("/process_question", response_model=OutModel|QA)
async def process_question(question: str = Query(..., title="User Question")):
    # print(question)
    split: QuestionSplit = InferLLM1().infer(question).output
    print(f"{split=}\n")

    if not isinstance(split, QuestionSplit):
        raise HTTPException(status_code=400, detail="Failed to split the question")

    if split.can_i_answer:
        answer = ask_llm(question)
        if not unanswered_regex.search(answer):
            print(f"{split=}")
            return QA(
                question=question,
                answer=answer,
            )
    
    function_docs = list(chain(*[vdb.similarity_search(task, k=3) for task in split['tasks']+[question, "llm"]]))
    # print(function_docs)
    functions = set([Function.model_validate(doc.metadata) for doc in function_docs])
    # print(f"{functions=}\n")
    input_schema = StepsInput(query=split['question'], steps=split['tasks'], functions=functions)
    
    step_output = InferLLM2().infer(input_schema)
    print(f"{step_output=}\n")

    if not isinstance(step_output, StepsOutput):
        raise HTTPException(status_code=400, detail="Failed to generate steps")

    runner = StepRunner(question, step_output.steps)
    runner.run_steps()
    print(f"{runner.context_dict=}\n")
    print(f"{list(runner.context_dict.values())[-1]=}")
    print(f"""{list(list(runner.context_dict.values())[-1].get("function", {"": {"output": ""}}).values())[-1]['output']=}""")
    try:
        response = OutModel(
            split=split,
            steps_input=input_schema, 
            steps_output=step_output, 
            context_dict=runner.context_dict, 
            response=list(list(runner.context_dict.values())[-1].get("function", {"": {"output": ""}}).values())[-1]['output'],
        )
    except ValidationError as exc:
        print(repr(exc.errors()[0]))
    filepath = Path(f"run_logs/{remove_special_chars(question.lower())}.json")
    filepath.mkdir(parents=True,exist_ok=True)
    with open(get_ts_filename(filepath), "w") as f:
        f.write(response.model_dump_json(indent=4))
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
