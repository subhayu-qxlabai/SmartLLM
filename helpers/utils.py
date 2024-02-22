import re
import json
import uuid
import traceback
from pathlib import Path
from itertools import chain
from datetime import datetime
from concurrent import futures
from typing import Callable, Iterable
from random import choices, randrange, shuffle,random

import tiktoken


SPECIAL_CHAR_REGEX = re.compile('[@#,.:;!?\s]')


def get_trace(e: Exception, n: int = 5):
    """Get the last n lines of the traceback for an exception"""
    return "".join(traceback.format_exception(e)[-n:])

def run_parallel_exec(exec_func: Callable, iterable: Iterable, *func_args, **kwargs):
    """
    Runs the `exec_func` function in parallel for each element in the `iterable` using a thread pool executor.
    
    Parameters:
        exec_func (Callable): The function to be executed for each element in the `iterable`.
        iterable (Iterable): The collection of elements for which the `exec_func` function will be executed.
        *func_args: Additional positional arguments to be passed to the `exec_func` function.
        **kwargs: Additional keyword arguments to customize the behavior of the function.
            - max_workers (int): The maximum number of worker threads in the thread pool executor. Default is 100.
            - quiet (bool): If True, suppresses the traceback printing for exceptions. Default is False.
    
    Returns:
        list[tuple]: A list of tuples where each tuple contains the element from the `iterable` and the result of executing the `exec_func` function on that element.

    Example:
        >>> from app.utils.helpers import run_parallel_exec
        >>> run_parallel_exec(lambda x: str(x), [1, 2, 3])
        [(1, '1'), (2, '2'), (3, '3')]
    """
    with futures.ThreadPoolExecutor(
        max_workers=kwargs.pop("max_workers", 100)
    ) as executor:
        # Start the load operations and mark each future with each element
        future_element_map = {
            executor.submit(exec_func, element, *func_args): element
            for element in iterable
        }
        result: list[tuple] = []
        for future in futures.as_completed(future_element_map):
            element = future_element_map[future]
            try:
                data = future.result()
            except Exception as exc:
                log_trace = exc if kwargs.pop("quiet", False) else get_trace(exc, 3)
                print(f"Got error while running parallel_exec: {element}: \n{log_trace}")
                result.append((element, exc))
            else:
                result.append((element, data))
        return result

def run_parallel_exec_but_return_in_order(exec_func: Callable, iterable: Iterable, *func_args, **kwargs):
    """
    Runs the `exec_func` function in parallel for each element in the `iterable` using a thread pool executor.
    Returns the result in the same order as the `iterable`.
    """
    # note this is usable only when iterable has types that are hashable
    result = run_parallel_exec(exec_func, iterable:=list(iterable), *func_args, **kwargs)

    # sort the result in the same order as the iterable
    result.sort(key=lambda x: iterable.index(x[0]))

    return [x[-1] for x in result]

def remove_special_chars(text: str):
    return SPECIAL_CHAR_REGEX.sub('', text)

def num_tokens_from_messages(messages: list[dict[str, str]], model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def get_openai_rate_limit_seconds(error_text: str) -> int:
    try:
        return int((re.findall(r"\s(\d+)\s(?:sec|min)?", error_text) or ['60'])[-1])
    except:
        return 60

def get_nested_value(d: dict, keys: list[str] | str):
    """
    Get nested value from dictionary
    :param d: Dictionary to get value from
    :param keys: List of keys or dot(`.`) separated string of keys
    :return: Value or None if not found
    """
    if isinstance(keys, str):
        keys = keys.split(".")
    if len(keys) == 1:
        return d.get(keys[0], None)
    return get_nested_value(d.get(keys[0], {}), keys[1:])
 
def set_nested_value(d: dict, keys: list[str] | str, value):
    """
    Set nested value in dictionary (inplace)
    :param d: Dictionary to set value in
    :param keys: List of keys or dot(`.`) separated string of keys
    :param value: Value to set
    """
    if isinstance(keys, str):
        keys = keys.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def remove_backticks(text: str) -> str:
    return re.sub(r"```\w+\n(.*)\n```", r"\1", text, flags=re.DOTALL)

def remove_comments(text: str) -> str:
    return re.sub(r'\s+//\s+.*', '', text)

def clean_json_str(text: str) -> str:
    return remove_comments(remove_backticks(text))

def get_ts_filename(filepath: str):
    filepath: Path = Path(filepath)
    filepath = filepath.parent / f"{filepath.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(random())[2:]}{filepath.suffix}"
    return filepath

def try_json_loads(text: str, default_return = []):
    try:
        return json.loads(text)
    except:
        if isinstance(default_return, list): 
            return [text]
        elif isinstance(default_return, dict):
            return {"data": text}
        elif isinstance(default_return, str):
            return text
        return default_return

def try_json_load(filename: str, default_return = []) -> dict | list:
    if not Path(filename).exists():
        return default_return
    return try_json_loads(open(filename, encoding="utf-8").read(), default_return)

def try_json_dumps(data: dict | list, indent=None):
    try:
        return json.dumps(data, indent=indent)
    except:
        return ""

def try_json_dump(data: dict | list, filename: str, indent=None):
    open(filename, "w", encoding="utf-8").write(try_json_dumps(data, indent))

def shuffle_list_of_dicts_based_on_a_key(ds: list[dict], min_len: int = 5, key: str = "can_i_answer") -> list[dict]:
    ds_true = [x for x in ds if x.get(key, True)]
    ds_false = [x for x in ds if not x.get(key, False)]
    min_len = min(len(ds_true), len(ds_false)) if not min_len else min_len

    ds_final = choices(ds_true, k=min_len) + choices(ds_false, k=min_len)
    shuffle(ds_final)
    return ds_final

def join_lists_from_dir(directory: str|Path) -> list[dict|str]:
    return list(chain(*[try_json_load(x) for x in Path(directory).rglob("*.json")]))

def get_all_questions(questions_dir: str|Path) -> list[str]:
    return join_lists_from_dir(questions_dir)

def get_random_questions(questions_dir: str|Path, k=10) -> list[str]:
    return choices(get_all_questions(questions_dir), k=k)

def get_all_splits(questions_dir: str|Path) -> list[dict]:
    return join_lists_from_dir(questions_dir)

def get_all_split_questions(questions_dir: str|Path) -> list[str]:
    return [x["question"] for x in get_all_splits(questions_dir) if isinstance(x, dict) and "question" in x]

def get_chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_json_list_from_string(s: str) -> list:
    clean_s = re.sub(r"^.*?(\[(.*)\]).*", r"\1", s, flags=re.DOTALL)
    return json.loads(clean_s)

def get_timestamp_uid(make_uuid=True, local_timezone=True):
    """Get a unique id for a timestamp. If `make_uuid` is True, an UUID will be generated from the timestamp."""
    if local_timezone:
        timestamp = datetime.now().isoformat()
    else:
        timestamp = datetime.utcnow().isoformat()
    uid = re.sub(r'[:\.\-\+TZ\s]', '', timestamp)
    if make_uuid:
        rndm = str(randrange(10 ** 11, 10 ** 12))
        uid = uuid.UUID(f'{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{rndm}')
    return uid

def convert_to_dict(val: str | dict | None) -> dict | None:
    if val is None:
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val

def convert_to_str(val: str | dict | None) -> str | None:
    if val is None:
        return None
    if isinstance(val, dict):
        return json.dumps(val)
    return val