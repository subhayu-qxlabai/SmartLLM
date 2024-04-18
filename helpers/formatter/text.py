from helpers.formatter.messages import MessagesFormatter


class TextFormatter(MessagesFormatter):
    def __init__(
        self,
        system_template: str = "<<SYS>> {system} <<SYS>>",
        user_template: str = "[INST] {user} [/INST]",
        assistant_template: str = "{assistant}",
        system_key: str = "system",
        user_key: str = "user",
        assistant_key: str = "assistant",
        separator: str = " ",
    ):
        self.system_key = system_key
        self.user_key = user_key
        self.assistant_key = assistant_key
        self._keys = [system_key, user_key, assistant_key]
        messages: list[list[dict[str, str]]] = []
        super().__init__(
            messages,
            system_template,
            user_template,
            assistant_template,
            system_key,
            user_key,
            assistant_key,
            separator,
            "role",
            "content",
        )
        
    def format_text(self, system: str = "", user: str = "", assistant: str = "") -> str:
        self.messages = [[
            {"role": self.system_key, "content": system},
            {"role": self.user_key, "content": user},
            {"role": self.assistant_key, "content": assistant},
        ]]
        return self.format().formatted_messages[-1]
    