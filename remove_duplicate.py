import json
from pathlib import Path
import pandas as pd
import numpy as np

directory_path = Path("eval_dataset_for_evaluation_metric")
split_file_path = directory_path / "gold_standards.json"
questions_file_path = directory_path / "questions.json"
with open(split_file_path, 'r+') as file:
    file.seek(0)
    existing_data = json.load(file)
    print(len(existing_data))
    # with open(questions_file_path,"r") as file2:
    #     file2.seek(0)
    #     existing_data2 = json.load(file2)
    #     df2=pd.DataFrame(existing_data2)
        # print(type(df2[0]))
        # print(df2[0])
        # df2_series=df2[0]
    # df1=ps.DataFrame(existing_data)
#     print(len(existing_data))
#     unique_data = list({json.dumps(item, sort_keys=True) for item in existing_data})
#     # print(unique_data)
#     unique_json_data = [json.loads(item) for item in unique_data]

        
    
    df = pd.DataFrame(existing_data)

    # The below commented code i replaced all the tasks values to the empty list for the cases of can_i_answer=True
    '''
    indices = df.index[df['can_i_answer'] == True].tolist()
    # empty_list = [[] for _ in indices]
    empty_lists_dict = {index: [] for index in indices}
    # df.loc[df['can_i_answer'] == True, 'tasks'] = empty_list
    for index, empty_list in empty_lists_dict.items():
        df.at[index, 'tasks'] = empty_list
    # df.loc[df['tasks'] == None, 'tasks'] = []
    tasks_series=df[df["can_i_answer"]==True]["tasks"]
    # print(tasks_series.to_list())
    file.seek(0)  # Move to the beginning of the file
    file.truncate()  # Clear the file contents
    df.to_json(file, orient='records')
    '''
    series=df[df["can_i_answer"]==True]["question"]
    # series=df[df["can_i_answer"]==True]
    print(series.count())
    # print(json.dumps(series["tasks"].to_list()))
    # print(json.dumps(series.to_dict(orient="records")))
    # not_in_series = [x for x in existing_data if x not in series.values]
    # not_in_df_series = df2_series[~df2_series.isin(series)]
    # print(not_in_df_series)
    # print(len(not_in_series))
    # print(df["question"])
    # df_unique = df.drop_duplicates('question')
    # unique_json_data = df_unique.to_dict(orient='records')
    # print(len(unique_json_data))
    # print(unique_json_data)
    # file.seek(0)
    # json.dump(unique_json_data, file)