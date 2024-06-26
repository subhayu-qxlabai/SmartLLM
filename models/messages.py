import json
from enum import Enum
from typing import Any
from pathlib import Path
from random import choice
from functools import partial
from typing_extensions import deprecated

from datasets import Dataset
from pydantic import root_validator

from helpers.utils import hash_uuid
from models.base import CustomBaseModel as BaseModel
from helpers.formatter import MessagesFormatter


class ConversationFormat(str, Enum):
    alpaca = "alpaca"
    openai = "openai"


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
    llm: str | None = None
    language: str = "english"
    messages: list[Message] = []

    def to_list(self) -> list[dict[str, str]]:
        return self.model_dump(mode="json")["messages"]

    @property
    def formatter(self):
        return partial(MessagesFormatter, messages=[self.to_list()])

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def __contains__(self, item):
        return item in self.messages

    def __hash__(self):
        return hash_uuid(
            self.llm or ""+self.language+"|".join([x.content for x in self.messages if x.role != "system"])
        ).int

    def __repr__(self):
        return self.model_dump_json()


class AlpacaMessages(BaseModel):
    llm: str | None = None
    language: str = "english"
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
            llm=self.llm,
            language=self.language,
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
    
    @property
    def hash_text(self):
        return "|".join((
            self.llm or "",
            self.language,
            # self.system or "",
            self.input or "",
            self.output or "",
        ))

    def __hash__(self):
        return hash_uuid(self.hash_text).int

    def __repr__(self):
        return self.model_dump_json()


class BaseMessagesList(BaseModel):
    messages_list: list[Messages | AlpacaMessages] = []

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

    @classmethod
    def from_dataset(cls, d: Dataset, llm: str | None = None):
        d_type: type[Messages | AlpacaMessages] = cls.model_fields[
            "messages_list"
        ].annotation.__args__[0]
        if set(d[0]) - set(d_type.model_fields) != set():
            return cls()
        if d_type == AlpacaMessages and llm is not None:
            return cls(
                messages_list=[
                    (
                        AlpacaMessages(llm=llm, **x)
                        if "llm" not in x
                        else AlpacaMessages(**x)
                    )
                    for x in d.to_list()
                ]
            )
        return cls(messages_list=[d_type(**x) for x in d.to_list()])

    @classmethod
    @deprecated("Use `.from_parquet` instead")
    def from_jsonl(cls, jsonl_file: str | Path, llm: str | None = None, verbose: bool = False):
        try:
            d = Dataset.from_json(str(jsonl_file))
        except Exception as e:
            if verbose:
                print(f"Error loading jsonl: {e}")
            return cls()
        return cls.from_dataset(d, llm)
    
    @classmethod
    def from_parquet(cls, parquet_file: str | Path, llm: str | None = None, verbose: bool = False):
        try:
            d = Dataset.from_parquet(str(parquet_file))
        except Exception as e:
            if verbose:
                print(f"Error loading parquet: {e}")
            return cls()
        return cls.from_dataset(d, llm)

    def unique(self):
        return self.__class__(
            messages_list=list({hash_uuid(x.hash_text): x for x in self.messages_list}.values())
        )

    def __add__(self, other: "BaseMessagesList"):
        if type(self) != type(other):
            raise NotImplementedError(f"Cannot add {type(self)} and {type(other)}")
        _messages_list = self.messages_list + other.messages_list
        return self.__class__(messages_list=_messages_list).fill_systems().unique()

    def fill_systems(self, systems: list[str] = None):
        _systems = {
            (
                x.system
                if isinstance(x, AlpacaMessages)
                else (
                    x[0].content
                    if isinstance(x, Messages) and len(x) >= 1 and x[0].role == "system"
                    else None
                )
            )
            for x in self.messages_list
        }
        if not isinstance(systems, list):
            systems = []
        systems = systems + list(_systems)
        systems = [x for x in systems if isinstance(x, str)]
        if len(systems) == 0:
            return self
        for x in self.messages_list:
            if isinstance(x, AlpacaMessages) and not x.system:
                x.system = choice(systems)
            elif isinstance(x, Messages) and len(x) >= 1 and x[0].role != "system":
                x.messages = [
                    Message(role="system", content=choice(systems))
                ] + x.messages
        return self.__class__(messages_list=self.messages_list)

    def to_dataset(self) -> Dataset:
        d = Dataset.from_list(self.model_dump(mode="json")["messages_list"])
        return d

    def to_jsonl(self, jsonl_file: str | Path):
        self.to_dataset().to_json(str(jsonl_file))

    def __repr__(self):
        return f"{self.__class__.__name__}(messages_list={len(self.messages_list)})"


class MessagesList(BaseMessagesList):
    messages_list: list[Messages] = []


class AlpacaMessagesList(BaseMessagesList):
    messages_list: list[AlpacaMessages] = []

    def to_openai(self) -> MessagesList:
        return MessagesList(messages_list=[x.to_messages() for x in self.messages_list])


def messages_factory(fmt: ConversationFormat):
    if fmt == ConversationFormat.alpaca:
        return AlpacaMessages
    elif fmt == ConversationFormat.openai:
        return Messages


def messages_list_factory(fmt: ConversationFormat):
    if fmt == ConversationFormat.alpaca:
        return AlpacaMessagesList
    elif fmt == ConversationFormat.openai:
        return MessagesList
