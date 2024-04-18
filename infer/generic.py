from infer.base import InferBase
from helpers.formatter.text import TextFormatter


class InferGeneric(InferBase):
    def __init__(self, formatter: TextFormatter = None, use_cache: bool = True):
        super().__init__(
            model_kwargs={"use_cache": use_cache},
            pretrained_model_name_or_path="bigscience/bloom-7b1",
        )
        self.formatter = formatter or TextFormatter(system_template="")

    def infer(self, request: str, include_system: bool = False):
        request_str: str = self.formatter.format_text(user=request)
        return self._infer(request_str)
