from infer.base import InferBase
from helpers.formatter.text import TextFormatter


class InferGeneric(InferBase):
    def __init__(self, formatter: TextFormatter = None):
        super().__init__(
            pretrained_model_name_or_path="bigscience/bloom-7b1",
        )
        self.formatter = formatter or TextFormatter()

    def infer(self, request: str, include_system: bool = False):
        request_str: str = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": request}
            ], 
            tokenize=False
        )
        return self._infer(request_str)
        