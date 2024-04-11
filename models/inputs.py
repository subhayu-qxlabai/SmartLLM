from typing import List
from models.base import CustomBaseModel as BaseModel
from helpers.utils import hash_uuid


class FunctionProperties(BaseModel):
    name: str
    type: str
    description: str

class Parameters(BaseModel):
    type: str = "object"
    properties: List[FunctionProperties|dict]
    required: List[str] = []

class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters

    def __hash__(self) -> int:
        return hash_uuid(f"{self.name}|{self.description}|{','.join([x.name for x in self.parameters.properties])}").int

class Step(BaseModel):
    query: str
    functions: List[Function]

class StepsInput(BaseModel):
    query: str
    steps: List[str]
    functions: List[Function]

    def __hash__(self) -> int:
        return hash_uuid(f"{self.query}").int
    