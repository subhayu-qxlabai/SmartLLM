import re
from typing import List, Optional

from pydantic import BaseModel, Field, root_validator


step_and_func_id = re.compile(r"step_([0-9]+)\.(\w+)\.(\w+_[0-9]+)")


def is_valid_path(path: str):
    # Basic validation
    return path.count(".") == 2 and path.startswith("step_")
    # Strict validation
    found = step_and_func_id.findall(path)
    return not (not found or len(found[0]) != 3)


def get_step_item_id(path: str) -> dict[str, int | str]:
    if not is_valid_path(path):
        return {}
    found = step_and_func_id.findall(path)
    return {
        "step_id": int(found[0][0]) - 1,
        "type": found[0][1],
        "type_id": found[0][2],
    }


class ParameterEntry(BaseModel):
    name: str
    value: str | int | float | list | dict


class FunctionEntry(BaseModel):
    id: str
    name: str
    parameters: List[ParameterEntry]

    @root_validator(pre=True)
    def validate_parameters(cls, values):
        values["parameters"] = [x for x in values.get("parameters") if x.get("value")]
        return values


class ExtractSchemaEntry(BaseModel):
    name: str
    type: str
    description: str


class ExtractEntryBase(BaseModel):
    id: str
    eschema: List[ExtractSchemaEntry] = Field(alias="schema")


class ExtractEntry(ExtractEntryBase):
    context: List[str]

    @property
    def valid_contexts(self):
        return [x for x in self.context if is_valid_path(x)]

    @property
    def context_dicts(self):
        return [get_step_item_id(x) for x in self.valid_contexts]


class ExtractEntryWithFunction(BaseModel):
    eschema: List[ExtractSchemaEntry] = Field(alias="schema")
    functions: List[FunctionEntry]


class FunctionWithExtract(BaseModel):
    function: FunctionEntry
    extract: ExtractEntryBase


class Step(BaseModel):
    id: str
    thought: str
    explore_tools: str
    available_tools: List[str]
    choose_tool: str
    understand_dependencies: Optional[str] = None
    extract: List[ExtractEntry] = []
    function: List[FunctionEntry]

    def get_function_by_name(self, name: str) -> FunctionEntry | None:
        return ([x for x in self.function if x.name == name] or [None])[0]

    def get_function_by_id(self, _id: str) -> FunctionEntry | None:
        return ([x for x in self.function if x.id == _id] or [None])[0]

    def get_extract_by_id(self, _id: str) -> ExtractEntry | None:
        return ([x for x in self.extract if x.id == _id] or [None])[0]


class StepsOutput(BaseModel):
    overview: str
    steps: List[Step]

    def get_item_by_context_path(self, path: str):
        ids = get_step_item_id(path)
        if not ids:
            return
        step_id = int(ids.get("step_id", 1_000_000))
        type_name = ids.get("type")
        type_id = ids.get("type_id")
        if step_id >= len(self.steps):
            return
        if type_name == "function":
            return self.steps[step_id].get_function_by_id(type_id)
        elif type_name == "extract":
            return self.steps[step_id].get_extract_by_id(type_id)

    def get_items_by_context_paths(self, paths: list[str]):
        return {x: self.get_item_by_context_path(x) for x in paths}

    def get_extracts_with_functions(self):
        extracts_with_functions: list[ExtractEntryWithFunction] = []
        for step in self.steps:
            if not step.extract:
                continue
            for extract in step.extract:
                context_dicts = self.get_items_by_context_paths(extract.context)
                functions = [
                    v for v in context_dicts.values() if isinstance(v, FunctionEntry)
                ]
                extract_w_func = ExtractEntryWithFunction(
                    schema=extract.schema,
                    functions=functions,
                )
                extracts_with_functions.append(extract_w_func)
        return extracts_with_functions

    def get_functions_with_extracts(self):
        func_w_extracts: list[FunctionWithExtract] = []
        for step in self.steps:
            if not step.extract:
                continue
            for extract in step.extract:
                context_dicts = self.get_items_by_context_paths(extract.context)
                context_functions = [
                    v for v in context_dicts.values() if isinstance(v, FunctionEntry)
                ]
                [
                    func_w_extracts.append(
                        FunctionWithExtract(
                            function=func,
                            extract=ExtractEntryBase.model_validate(
                                extract.model_dump()
                            ),
                        )
                    )
                    for func in context_functions
                ]
        return func_w_extracts

    def __hash__(self):
        return hash(f"{self.overview}")
