from infer.base import InferBase
from helpers.formatter.text import TextFormatter


class InferGeneric(InferBase):
    def __init__(self, formatter: TextFormatter = None, use_cache: bool = True):
        super().__init__(
            pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
            hf_token="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",
            model_kwargs={"use_cache": use_cache},
        )
        self.formatter = formatter or TextFormatter()

    def infer(self, request: str, include_system: bool = False):
        request_str: str = self.formatter.format_text(
            system="You are a helpful assistant.", 
            user=request
        )
        return self._infer(request_str)
