import json
from functools import partial
from typing import Any

from pydantic import BaseModel, root_validator
from helpers.formatter import MessagesFormatter

def json_dumps_or_str(data):
    if isinstance(data, str):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump_json()
    try:
        return json.dumps(data)
    except:
        return str(data)

class Message(BaseModel):
    role: str
    content: str

    @root_validator(pre=True)
    def validate(cls, values: dict):
        values["content"] = json_dumps_or_str(values["content"])
        return values

class SystemMessage(Message):
    role: str = "system"

class UserMessage(Message):
    role: str = "user"

class AssistantMessage(Message):
    role: str = "assistant"


class Messages(BaseModel):
    messages: list[Message] = []

    def to_list(self) -> list[dict[str, str]]:
        return self.model_dump(mode="json")["messages"]
    
    @property
    def formatter(self):
        return partial(MessagesFormatter, messages=[self.to_list()])

    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index):
        return self.messages[index]
    
    def __iter__(self):
        return iter(self.messages)
    
    def __contains__(self, item):
        return item in self.messages
    
    def __repr__(self):
        return self.model_dump_json()
    

class AlpacaMessages(BaseModel):
    system: str | None = None
    input: str | None = None
    output: str | None = None
    
    @root_validator(pre=True)
    def validate(cls, values: dict[str, Any]):
        for k, v in values.items():
            if v is not None:
                values[k] = json_dumps_or_str(v)
        return values
    
    def to_messages(self) -> Messages:
        return Messages(
            messages=[
                SystemMessage(content=self.system),
                UserMessage(content=self.input),
                AssistantMessage(content=self.output),
            ]
        )
    
    def to_list(self) -> list[dict[str, str]]:
        return self.to_messages().to_list()
    
    @property
    def formatter(self):
        return partial(MessagesFormatter, messages=[self.to_messages().to_list()])

    def __repr__(self):
        return self.model_dump_json()
    

class MessagesList(BaseModel):
    messages_list: list[Messages] = []
    
    def to_list(self) -> list[list[dict[str, str]]]:
        return [x.to_list() for x in self.messages_list]
    
    @property
    def formatter(self):
        return partial(MessagesFormatter, messages=self.to_list())

    def __len__(self):
        return len(self.messages_list)
    
    def __getitem__(self, index):
        return self.messages_list[index]
    
    def __iter__(self):
        return iter(self.messages_list)
    
    def __contains__(self, item):
        return item in self.messages_list
    
    def __repr__(self):
        return f"{self.__class__.__name__}(messages_list={len(self.messages_list)})"
    
    