import json
from pathlib import Path
import pandas as pd

directory_path = Path("eval_dataset_for_evaluation_metric")
eval_file_path = directory_path / "eval_dataset.json"



def drop_empty_rows(path:Path):
    with open(path, "r+") as eval_file:
        dict_json = json.load(eval_file)
        df=pd.DataFrame(dict_json)
        df=df.dropna()
        dict_json=df.to_dict(orient="records")
        # print(dict_json)
        # print(df.count())
        eval_file.seek(0)
        json.dump(dict_json, eval_file)
        eval_file.truncate()