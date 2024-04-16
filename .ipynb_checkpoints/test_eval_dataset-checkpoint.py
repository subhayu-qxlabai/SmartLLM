
from pathlib import Path
import json
import pandas as pd
from helpers.text_utils import TextUtils

directory_path = Path("eval_dataset_for_evaluation_metric")
eval_dataset_path = directory_path / "eval_dataset.json"
new_eval_dataset_path = directory_path / "new_eval_dataset.json"

# with open(eval_dataset_path) as que_file:
#     dict_json = json.load(que_file)
#     language_df=pd.Dataframe(dict_json)


def stringify_expected_output(value):
    res=json.dumps(value)
    return res

def change_actualoutput(value):
    # start_string="<<SYS>>"
    # end_string="[/INST]"
    # prompt=TextUtils.get_middle_text(value, start_string, end_string).strip()
    # full_prompt=start_string+prompt+end_string
    start_string=' {"output": "'
    end_string='"}</s>'
    
    response = TextUtils.get_middle_text(value, start_string, end_string).strip()
    response = response.replace('\\"', '"')
    return response



df=pd.read_json(new_eval_dataset_path)
# df = df[df["actual_output"] != '']
# csv_path=directory_path / "empty_actual_output.csv"
# df[["language_type","questions","expected_output"]].to_csv(csv_path,columns=["language_type","questions","expected_output"],index_label="id")
df["expected_output"]=df["expected_output"].apply(stringify_expected_output)
json_data = df.to_json(orient='records')

with open(new_eval_dataset_path, 'w') as f:
    f.seek(0)
    f.write(json_data)
    f.truncate()
