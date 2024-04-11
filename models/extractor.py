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


class ExtractorIO(BaseModel):
    input: ExtractorInput
    output: dict[str, Any]


class ExtractorDatasetRowOutput(BaseModel):
    function_output: dict[str, Any] = None
    extracted_schema: dict[str, Any] = None

    def to_extractor_io(self, input_schema: list[ExtractSchemaEntry | dict]):
        input_schema = [
            x if isinstance(x, dict) else x.model_dump() for x in input_schema
        ]
        return ExtractorIO(
            input=ExtractorInput(
                schema=input_schema,
                context=list(self.function_output.values()),
            ),
            output=self.extracted_schema,
        )
