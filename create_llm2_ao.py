from pathlib import Path
import json
import pandas as pd
from models.inputs import StepsInput
from models.outputs import StepsOutput

directory_path = Path("eval_dataset_for_evaluation_metric")
llms2_json = directory_path / "llm2_eval_dataset.json"

df=pd.read_json(llms2_json)


from infer import InferLLM2
def run_llm2_actual_output(value):
    if value==None:
        step_output=""
        return step_output
    # print(type(value))
    input_schema = StepsInput(**json.loads(value))
    try:
        step_output=InferLLM2().infer(input_schema)
        return step_output.model_dump_json()
    # print(type(step_output))
    except Exception as e:
        print(e)
        step_output=""
        return step_output

df = df.replace('', pd.NA).dropna()
df["actual_output"]=df["actual_input"].apply(run_llm2_actual_output)
df = df.replace('', pd.NA).dropna()
json_data=df.to_json(orient="records")
with open(llms2_json, 'w') as f:
    f.seek(0)
    f.write(json_data)
    f.truncate()
