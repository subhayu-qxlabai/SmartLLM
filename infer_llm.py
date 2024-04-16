# Run with: INFER_TYPE=<infer_type> python infer_llm.py
# INFER_TYPE can be one of: llm1, llm2, llm3 or generic

import os
from fastapi import FastAPI, Body

from infer import get_infer, InferType
from helpers.middleware import ProcessTimeMiddleware
from infer.llm1 import SplitOutput
from infer.llm2 import StepsInput, StepsOutput
from infer.llm3 import ExtractorInput, Any


infer_type = InferType.from_substr(os.getenv("INFER_TYPE", ""))
if infer_type is None:
    raise ValueError(f"INFER_TYPE must be one of {', '.join([x.value for x in InferType])}")

app = FastAPI(title=f"Infer{infer_type.name}")
app.add_middleware(ProcessTimeMiddleware)

REQUEST_TYPE = str|StepsInput|ExtractorInput
RESPONSE_TYPE = SplitOutput|StepsOutput|dict[str, Any]|str


@app.get("/infer", response_model=RESPONSE_TYPE)
async def infer(request: REQUEST_TYPE = Body(...), include_system: bool = True):
    infer_fn = get_infer(infer_type, use_cache=False)
    return infer_fn(request, include_system)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("infer_llm:app", host="0.0.0.0", port=8080, reload=True)
