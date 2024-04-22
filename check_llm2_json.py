from pathlib import Path
import numpy as np

directory_path = Path("eval_dataset_for_evaluation_metric")
questions_file_path = directory_path / "llm2_eval_dataset.json"

import pandas as pd

json_data=pd.read_json(questions_file_path)

#print(json_data.head(2))

#null_rows = json_data.loc[json_data["actual_input"].isnull()]
df = json_data.actual_input.replace('',np.nan)
json_data["actual_input"]=df

# print(json_data.count())
# null_rows=json_data[json_data['actual_input']==""]
# print(null_rows)
