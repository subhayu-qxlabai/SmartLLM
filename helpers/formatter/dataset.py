from datasets import Dataset

from helpers.formatter.messages import MessagesFormatter

class DatasetFormatter(MessagesFormatter):
    def __init__(
        self,
        dataset: Dataset,
        messages_column: str = "messages",
        system_template: str = "<<SYS>> {system} <<SYS>>",
        user_template: str = "[INST] {user} [/INST]",
        assistant_template: str = "{assistant}",
        system_key: str = "system",
        user_key: str = "user",
        assistant_key: str = "assistant",
        separator: str = " ",
        message_role_field: str = "role",
        message_content_field: str = "content",
    ):
        assert messages_column in dataset.column_names, f"{messages_column!r} not found in dataset"
        super().__init__(
            dataset[messages_column],
            system_template,
            user_template,
            assistant_template,
            system_key,
            user_key,
            assistant_key,
            separator,
            message_role_field,
            message_content_field,
        )