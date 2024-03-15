from typing import Any
from pydantic import Field
from models.base import CustomBaseModel as BaseModel


class ExtractSchemaEntry(BaseModel):
    name: str
    type: str
    description: str

class ExtractorInput(BaseModel):
    eschema: list[ExtractSchemaEntry] = Field([], alias="schema")
    context: list[Any]

