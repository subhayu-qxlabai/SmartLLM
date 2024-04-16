from enum import Enum
from typing import NamedTuple

from helpers.formatter.text import TextFormatter
from infer.base import InferBase
from infer.llm1 import InferLLM1, SplitOutput
from infer.llm2 import InferLLM2, StepsInput, StepsOutput
from infer.llm3 import InferLLM3, ExtractorInput, Any
from infer.generic import InferGeneric


class InferType(str, Enum):
    LLM1 = "llm1"
    LLM2 = "llm2"
    LLM3 = "llm3"
    GENERIC = "generic"
    
    @classmethod
    def from_substr(cls, s: str, none_on_fail: bool = True):
        s = str(s)
        name = ([x.name for x in cls if s.find(x.name) != -1] or [None])[0]
        value = ([x.value for x in cls if s.find(x.value) != -1] or [None])[0]
        if name:
            return cls[name]
        if value:
            return cls(value)
        if not none_on_fail:
            raise ValueError(f"Failed to find LLMType from string: {s}")


class InferData(NamedTuple):
    model: InferBase
    input: type
    output: type


def get_infer(infer_type: InferType|str, formatter: TextFormatter = None, use_cache: bool = True):
    if infer_type in [InferType.LLM1, InferType.LLM1.value]:
        return InferLLM1(formatter, use_cache).infer
    elif infer_type in [InferType.LLM2, InferType.LLM2.value]:
        return InferLLM2(formatter, use_cache).infer
    elif infer_type in [InferType.LLM3, InferType.LLM3.value]:
        return InferLLM3(formatter, use_cache).infer
    elif infer_type in [InferType.GENERIC, InferType.GENERIC.value]:
        return InferGeneric(formatter, use_cache).infer
    else:
        raise ValueError(f"Invalid model name: {infer_type}")

def get_infer_data(infer_type: InferType|str):
    if infer_type in [InferType.LLM1, InferType.LLM1.value]:
        return InferData(InferLLM1, str, SplitOutput|str)
    elif infer_type in [InferType.LLM2, InferType.LLM2.value]:
        return InferData(InferLLM2, StepsInput, StepsOutput|str)
    elif infer_type in [InferType.LLM3, InferType.LLM3.value]:
        return InferData(InferLLM3, ExtractorInput, dict[str, Any])
    elif infer_type in [InferType.GENERIC, InferType.GENERIC.value]:
        return InferData(InferGeneric, str, str)
    else:
        raise ValueError(f"Invalid model name: {infer_type}")
