from evaluate import load
from infer.question_breaker import get_llm_response

squad_metric = load("squad")


input={"question":"\u00bfExisten incentivos para la instalaci\u00f3n de estaciones de carga de veh\u00edculos el\u00e9ctricos en Italia en 2023?"}

# input=json.dumps(input).strip()

prompt_template = f"""<<SYS>> Being an honest and smart assistant talented in breaking down questions into actionable items, you're charged with interpreting a JSON-formatted question. Your output must be a JSON object articulated with two keys: can_i_answer (indicating true if the inquiry is answerable using internal capabilities, or false if it requires external resources) and tasks, delineating the series of steps to answer the question with external aids if can_i_answer is false. <<SYS>> [INST] {input} [/INST] """

output={"question":"\u00bfExisten incentivos para la instalaci\u00f3n de estaciones de carga de veh\u00edculos el\u00e9ctricos en Italia en 2023?","tasks":["RESEARCH: Investigate whether there are incentives for installing electric vehicle charging stations in Italy in 2023"],"can_i_answer":False}

actual_output=get_llm_response(prompt_template)

predictions = [{'prediction_text': prompt_template, 'id': '56e10a3be3433e1400422b22'}]
references = [{'answers': actual_output, 'id': '56e10a3be3433e1400422b22'}]
results = squad_metric.compute(predictions=predictions, references=references)