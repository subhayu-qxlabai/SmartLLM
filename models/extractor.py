from typing import Any
from pydantic import BaseModel


class ExtractSchemaEntry(BaseModel):
    name: str
    type: str
    description: str

class ExtractorInput(BaseModel):
    schema: list[ExtractSchemaEntry]
    context: list[Any]

