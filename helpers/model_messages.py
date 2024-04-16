from transformers import AutoTokenizer

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

class ModelMessages:
    def __init__(self, mode_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name)
    
    def model_format_single(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            print(e)
            return self.tokenizer.apply_chat_template(messages[1:], tokenize=False)

    def format_dataset_messages(self, messages_dict: dict[str, list[dict[str, str]]], key: str = "messages"):
        messages = messages_dict.get(key, [])
        return {key: self.model_format_single(messages)}

    def model_format(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        try:
            self.model_format_single(messages_list[0])
        except Exception as e1:
            try:
                self.model_format_single(messages_list[0][1:])
                messages_list=[x[1:] for x in messages_list]
            except Exception as e2:
                print(e1)
                print(e2)
                return []
        return [self.model_format_single(x) for x in messages_list]
