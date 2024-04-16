import re
import json
from itertools import chain
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks

from helpers.vectorstore.faisser import FaissDB
from models.generic import QuestionSplit
from models.inputs import StepsInput, Function
from models.outputs import StepsOutput
from step_runner import StepRunner
from infer.generic import ask_llm
from infer.steps_generator import get_steps
from infer.question_breaker import break_question
from helpers.middleware import ProcessTimeMiddleware
from helpers.utils import get_ts_filename, remove_special_chars
from pathlib import Path
import uuid
from uuid import UUID

app = FastAPI()
app.add_middleware(ProcessTimeMiddleware)

vdb = FaissDB(filename="/workspace/New_code/SmartLLM/functions.pkl")

unanswered_regex = re.compile(
    r"(as\s*of\s*my\s*last|as\s*of\s*my\s*current|last\s*knowledge|knowledge.*\d{4})", 
    flags=re.IGNORECASE | re.DOTALL,
)

tasks = {}

class QA(BaseModel):
    question: str
    answer: str

class OutModel(BaseModel):
    split: QuestionSplit | None = None
    steps_input: StepsInput | str | None = None
    steps_output: StepsOutput | str | None = None
    context_dict: dict | None = None
    response: str | None = None
    
# , response_model=OutModel|QA

# @app.post("/start-task/")
# @app.get("/process_question")@app.get("/process_question")
# async def start_task(background_tasks: BackgroundTasks, question: str = Query(..., title="User Question")):
#     task_id = str(uuid.uuid4())
#     tasks[task_id] = "in progress"
#     background_tasks.add_task(process_question, task_id, question)
    
#     return {"task_id": task_id}
    # return 


# @app.get("/task-status/")
# async def task_status(task_id: str):
    
#     if task_id not in tasks:
#         return {"status": "unknown"}
#     else:
#         return tasks[task_id]
        # return {"status": tasks[task_id]["status"], "Output": tasks[task_id]["output"]}
    

@app.get("/process_question", response_model=OutModel|QA)
async def process_question(question: str = Query(..., title="User Question")):
    print(question)
    split: QuestionSplit = break_question(question)
    print(f"{split=}\n")

    # if not isinstance(split, QuestionSplit):
    #     raise HTTPException(status_code=400, detail="Failed to split the question")

    if split["can_i_answer"]:
        
        answer = ask_llm(question)

        if not unanswered_regex.search(answer):
            print(f"{split=}")
            # tasks[task_id] = {"status": "completed", "output": answer}

            return QA(question= question,
                      answer= answer   
            )

    print(split["tasks"]+[question, "llm"])
    function_docs = list(chain(*[vdb.similarity_search(task, k=3) for task in split["tasks"]+[question, "llm"]]))
    # print(function_docs)
    functions = set([Function.model_validate(doc.metadata) for doc in function_docs])
    # print(f"{functions=}\n")
    input_schema = StepsInput(query=split["question"], steps=split["tasks"], functions=functions)
    
    step_output: StepsOutput = get_steps(input_schema)
    print(f"{step_output=}\n")

    if not isinstance(step_output, StepsOutput):
        raise HTTPException(status_code=400, detail="Failed to generate steps")

    runner = StepRunner(question, step_output["steps"])
    runner.run_steps()
    print(f"{runner.context_dict=}\n")
    print(f"{list(runner.context_dict.values())[-1]=}")
    print(f"""{list(list(runner.context_dict.values())[-1].get("function", {"": {"output": ""}}).values())[-1]['output']=}""")
    try:
        response = OutModel(
            splt=split,
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

    # tasks[task_id] = {"status": "completed", "output": response}
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main-Copy1:app", host="0.0.0.0", port=8080, reload=True)
