import json
import uuid
import random
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, root_validator

from helpers.utils import get_timestamp_uid
from models.inputs import StepsInput
from models.outputs import StepsOutput
from models.extractor import ExtractorInput
from models.generic import Question, QuestionSplit
from models.messages import (
    MessagesList, 
    Messages, 
    SystemMessage, 
    UserMessage, 
    AssistantMessage, 
    AlpacaMessages
)


UID_FUNCTION = lambda: get_timestamp_uid(make_uuid=True, local_timezone=True)
DEFAULT_DATASET_DIR = Path("generated/dataset")


def try_load_model(model: BaseModel, val):
    try:
        if isinstance(val, model):
            return val
        elif isinstance(val, str):
            return model.model_validate_json(val)
        elif isinstance(val, dict):
            return model.model_validate(val)
    except Exception as e:
        return val

class LLMType(str, Enum):
    LLM1 = "llm1"
    LLM2 = "llm2"
    LLM3 = "llm3"


class DatasetRow(BaseModel):
    uid: uuid.UUID | str = Field(default_factory=UID_FUNCTION)
    llm: LLMType
    system: str | None = None
    input: str | dict | None = None
    output: str | dict | None = None

    def to_file(self, dir: str | Path = DEFAULT_DATASET_DIR):
        dump_file = Path(dir) / self.llm.value / f"{str(self.uid)}.json"
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_file, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
        return dump_file

    @classmethod
    def from_file(cls, file: str | Path, log_errors=True):
        try:
            with open(file, "r") as f:
                return cls.model_validate_json(f.read())
        except Exception as e:
            if log_errors:
                print(e)

    @classmethod
    def from_dir(cls, dir: str | Path = DEFAULT_DATASET_DIR, log_errors=True):
        dir = Path(dir)
        dump_files = list(dir.rglob("*.json"))
        rows = [cls.from_file(file, log_errors) for file in dump_files]
        return [row for row in rows if row is not None]
    
    def to_alpaca(self):
        if self.system is None and self.input is None and self.output is None:
            return None
        return AlpacaMessages(
            system=self.system,
            input=self.input,
            output=self.output,
        )
    
    @classmethod
    def try_model_validate(cls, val, verbose: bool = True, none_on_fail: bool = False):
        try:
            return cls.model_validate(val)
        except Exception as e:
            if verbose:
                print(f"Failed to load model: {e}")
            return None if none_on_fail else val
    
    def __hash__(self):
        s = ""
        if self.system:
            s += str(self.system)
        if self.input:
            s += str(self.input)
        if self.output:
            s += str(self.output)
        if not s:
            s = str(self.uid)
        return hash(s)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(llm={self.llm}, system={self.system}, input={self.input}, output={self.output})"


class LLM1DatasetRow(DatasetRow):
    llm: LLMType = LLMType.LLM1
    input: Question
    output: QuestionSplit | None
    
    @root_validator(pre=True)
    def validate(cls, values: dict|BaseModel):
        if isinstance(values, BaseModel):
            values = values.model_dump()
        values["input"] = try_load_model(Question, values["input"])
        values["output"] = try_load_model(QuestionSplit, values["output"])
        return values


class LLM2DatasetRow(DatasetRow):
    llm: LLMType = LLMType.LLM2
    input: StepsInput | None
    output: StepsOutput | None
    
    @root_validator(pre=True)
    def validate(cls, values: dict|BaseModel):
        if isinstance(values, BaseModel):
            values = values.model_dump()
        values["input"] = try_load_model(StepsInput, values["input"])
        values["output"] = try_load_model(StepsOutput, values["output"])
        return values


class LLM3DatasetRow(DatasetRow):
    llm: LLMType = LLMType.LLM3
    input: ExtractorInput | None
    
    @root_validator(pre=True)
    def validate(cls, values: dict|BaseModel):
        if isinstance(values, BaseModel):
            values = values.model_dump()
        values["input"] = try_load_model(ExtractorInput, values["input"])
        return values


def llm_row_factory(llm_type: LLMType):
    match llm_type:
        case LLMType.LLM1:
            return LLM1DatasetRow
        case LLMType.LLM2:
            return LLM2DatasetRow
        case LLMType.LLM3:
            return LLM3DatasetRow
        case _:
            return DatasetRow


class LLMDatasetBase(BaseModel):
    rows: list[DatasetRow] = []

    def to_dir(self, dir: str | Path = DEFAULT_DATASET_DIR):
        return [row.to_file(dir) for row in self.rows]
    
    def get_llm(self, llm_type: LLMType):
        return self.__class__(rows=[row for row in self.rows if row.llm == llm_type])

    @classmethod
    def from_dir(cls, dir: str | Path = DEFAULT_DATASET_DIR, log_errors=True):
        return cls(rows=DatasetRow.from_dir(dir, log_errors))

    def to_messages(self):
        m_list = MessagesList()
        for row in self.rows:
            m = Messages()
            if row.system:
                m.messages.append(SystemMessage(content=row.system))
            if row.input:
                m.messages.append(UserMessage(content=row.input))
            if row.output and len(m) >= 1:
                m.messages.append(AssistantMessage(content=row.output))
            if len(m) >= 2:
                m_list.messages_list.append(m)
        return m_list
    
    def to_alpaca(self):
        alpaca_rows = [row.to_alpaca() for row in self.rows]
        return [row for row in alpaca_rows if row is not None]
    
    def to_file(self, file: str | Path = "generated.json", dir: str | Path = DEFAULT_DATASET_DIR):
        print(f"Dumped {len(self.rows)} rows to {(Path(dir) / file).absolute().as_posix()!r}")
        with open(Path(dir) / file, "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def from_file(cls, file: str | Path = "generated.json", dir: str | Path = DEFAULT_DATASET_DIR, log_errors=True):
        with open(Path(dir) / file, "r") as f:
            try:
                return cls.model_validate_json(f.read())
            except Exception as e:
                if log_errors:
                    print(f"Got error: {e}")
    
    def fill_systems(self, systems: list[str]):
        for row in self.rows:
            if not row.system:
                row.system = random.choice(systems)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rows={len(self.rows)})"
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, index):
        return self.__class__(rows=self.rows[index])
    
    def __iter__(self):
        return iter(self.rows)
    
    def __contains__(self, item):
        return item in self.rows
    
    def __add__(self, other: "LLMDatasetBase"):
        return self.__class__(rows=self.rows + other.rows)

class LLMDatasetWithTypes(LLMDatasetBase):
    rows: list[DatasetRow|LLM1DatasetRow|LLM2DatasetRow|LLM3DatasetRow] = []

class LLMDataset(LLMDatasetBase):
    rows: list[DatasetRow] = []

    def get_llm_type_rows(self, llm_type: LLMType = None, verbose: bool = False):
        if llm_type:
            dataset = self.get_llm(llm_type)
        else:
            dataset = self
        rows = [
            llm_row_factory(row.llm).try_model_validate(
                row.model_dump(mode="json"), none_on_fail=True, verbose=verbose
            )
            for row in dataset.rows
        ]
        return LLMDatasetWithTypes(rows=[row for row in rows if row is not None])
        
