from infer.llm1 import InferLLM1
from infer.llm2 import InferLLM2
from infer.llm3 import InferLLM3
from infer.generic import InferGeneric, TextFormatter
from enum import Enum

class InferType(str, Enum):
    LLM1 = "llm1"
    LLM2 = "llm2"
    LLM3 = "llm3"
    GENERIC = "generic"


def get_infer(infer_type: InferType|str):
    if infer_type in [InferType.LLM1, InferType.LLM1.value]:
        return InferLLM1
    elif infer_type in [InferType.LLM2, InferType.LLM2.value]:
        return InferLLM2
    elif infer_type in [InferType.LLM3, InferType.LLM3.value]:
        return InferLLM3
    elif infer_type in [InferType.GENERIC, InferType.GENERIC.value]:
        return InferGeneric
    else:
        raise ValueError(f"Invalid model name: {infer_type}")