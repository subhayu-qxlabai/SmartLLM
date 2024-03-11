import pandas as pd

from helpers.formatter.messages import MessagesFormatter


class AlpacaFormatter(MessagesFormatter):
    def __init__(
        self,
        messages: list[dict[str, str]],
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
        self.system_key = system_key
        self.user_key = user_key
        self.assistant_key = assistant_key
        self._keys = [system_key, user_key, assistant_key]
        messages: list[list[dict[str, str]]] = self._alpaca_to_messages(messages)
        super().__init__(
            messages,
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
        
    def _alpaca_to_messages(self, messages: list[dict[str, str]]):
        return [
            [
                {"role": key, "content": m.get(key)}
                for key in self._keys if key in m
            ]
            for m in messages 
            if m.get(self.user_key) and m.get(self.assistant_key)
        ]