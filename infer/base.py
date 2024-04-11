from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from helpers.text_utils import TextUtils

class InferBase:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: AutoTokenizer = None,
        model: AutoModelForCausalLM = None,
        common_kwargs: dict = {
            "device_map": 'auto', 
        },
        tokenizer_kwargs: dict = {
            "use_cache": False, 
        },
        model_kwargs: dict = {},
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, **common_kwargs, **tokenizer_kwargs
            )
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model_name_or_path, **common_kwargs, **model_kwargs
            )
        else:
            self.model = model

    def infer(
        self, 
        input_text: str, 
        encoder_kwargs: dict = {
            "return_tensors": "pt",
            "add_special_tokens": True,
        },
        decoder_kwargs: dict = {},
        generator_kwargs: dict = {
            "max_new_tokens": 8096,
            "do_sample": False,
        },
    ) -> str:
        encoded_input = self.tokenizer(input_text, **encoder_kwargs)
        model_inputs = encoded_input.to("cuda")
        generated_ids = self.model.generate(
            **model_inputs, pad_token_id=self.tokenizer.eos_token_id, **generator_kwargs
        )
        decoded_output = self.tokenizer.batch_decode(generated_ids, **decoder_kwargs)
        decoded_output: str = decoded_output[0]
        return TextUtils.get_middle_text(decoded_output, input_text, self.tokenizer.eos_token)

    def __call__(self, input_text: str, *args, **kwargs):
        return self.infer(input_text, *args, **kwargs)
