from functools import partial
from itertools import chain
import uuid
import random
from enum import Enum
from pathlib import Path
from pydantic import Field, root_validator
from datasets import Dataset
from tqdm import tqdm


from config import SUPPORTED_LANGUAGES
from helpers.utils import get_timestamp_uid, try_json_loads, run_parallel_exec, hash_uuid
from models.base import CustomBaseModel as BaseModel
from models.inputs import StepsInput
from models.outputs import StepsOutput
from models.extractor import ExtractorInput
from models.generic import Question, QuestionSplit
from models.messages import (
    AlpacaMessagesList,
    MessagesList, 
    Messages, 
    SystemMessage, 
    UserMessage, 
    AssistantMessage, 
    AlpacaMessages,
    ConversationFormat,
)


UID_FUNCTION = lambda: get_timestamp_uid(make_uuid=True, local_timezone=True)
UID_FUNCTION = uuid.uuid4
DEFAULT_DATASET_DIR = Path("generated/dataset")


def try_load_model(model: BaseModel, val):
    try:
        if isinstance(val, model):
            return model.model_validate(val)
        elif isinstance(val, str):
            return model.model_validate_json(val)
        elif isinstance(val, dict):
            return model.model_validate(val)
    except Exception as e:
        return val
    
def get_language(path: str | Path):
    return ([l for l in SUPPORTED_LANGUAGES if l in Path(path).as_posix()] or [None])[0]

class LLMType(str, Enum):
    LLM1 = "llm1"
    LLM2 = "llm2"
    LLM3 = "llm3"

    @classmethod
    def from_substr(cls, s: str, none_on_fail: bool = True):
        name = ([x.name for x in cls if s.find(x.name) != -1] or [None])[0]
        value = ([x.value for x in cls if s.find(x.value) != -1] or [None])[0]
        if name:
            return cls[name]
        if value:
            return cls(value)
        if not none_on_fail:
            raise ValueError(f"Failed to find LLMType from string: {s}")


class DatasetRow(BaseModel):
    uid: uuid.UUID | str = Field(default_factory=UID_FUNCTION)
    llm: LLMType
    language: str = "english"
    system: str | None = None
    input: str | dict | None = None
    output: str | dict | None = None
    
    @property
    def hash_text(self):
        return "|".join((
            self.llm.value, 
            self.language,
            # self.system, 
            str(self.input), 
            str(self.output), 
        ))
    
    @property
    def uid(self):
        return hash_uuid(self.hash_text)

    def to_file(self, dir: str | Path = DEFAULT_DATASET_DIR):
        dump_file = Path(dir) / self.llm.value / f"{str(self.uid)}.json"
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.write_text(self.model_dump_json())
        return dump_file

    @classmethod
    def from_file(cls, file: str | Path, log_errors = True, language: str = None):
        try:
            m = cls.model_validate_json(Path(file).read_text())
            default_lang = DatasetRow.model_fields['language'].default
            if language not in [None, default_lang] and m.language == default_lang:
                m.language = language
            return m
        except Exception as e:
            if log_errors:
                print(e)

    @classmethod
    def from_dir(cls, dir: str | Path = DEFAULT_DATASET_DIR, log_errors=True):
        dir = Path(dir)
        language = get_language(dir)
        dump_files = list(dir.rglob("*.json"))
        rows = [cls.from_file(file, log_errors, language) for file in dump_files]
        return [row for row in rows if row is not None]
    
    def to_alpaca(self):
        if self.system is None and self.input is None and self.output is None:
            return None
        return AlpacaMessages(
            llm=self.llm.value,
            language=self.language,
            system=self.system,
            input=self.input,
            output=self.output,
        )
    
    @classmethod
    def try_model_validate(cls, val, verbose: bool = True, none_on_fail: bool = False):
        try:
            m: cls = try_load_model(cls, val)
            return m
        except Exception as e:
            if verbose:
                print(f"Failed to load model: {e}")
            return None if none_on_fail else val
    
    def __hash__(self):
        return self.uid.int
    
    def __repr__(self):
        return f"{self.__class__.__name__}(llm={self.llm}, language={self.language}, system={self.system}, input={self.input}, output={self.output})"


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
    output: dict | None
    
    @root_validator(pre=True)
    def validate(cls, values: dict|BaseModel):
        if isinstance(values, BaseModel):
            values = values.model_dump()
        values["input"] = try_load_model(ExtractorInput, values["input"])
        values["output"] = try_json_loads(values["output"], default_return={})
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
    
    @property
    def languages(self):
        return set([row.language for row in self.rows])

    def to_dir(self, dir: str | Path = DEFAULT_DATASET_DIR):
        return [row.to_file(dir) for row in self.rows]
    
    def get_llm(self, llm_type: LLMType):
        return self.__class__(rows=[row for row in self.rows if row.llm == llm_type])

    @classmethod
    def from_dir(cls, dir: str | Path = DEFAULT_DATASET_DIR, log_errors=True):
        return cls(rows=DatasetRow.from_dir(dir, log_errors))

    def to_messages(self, fmt: ConversationFormat = ConversationFormat.alpaca):
        if fmt in [ConversationFormat.alpaca, ConversationFormat.alpaca.value]:
            alpaca_rows = [row.to_alpaca() for row in self.rows]
            return AlpacaMessagesList(
                messages_list=[row for row in alpaca_rows if row is not None]
            )
        elif fmt in [ConversationFormat.openai, ConversationFormat.openai.value]:
            m_list = MessagesList()
            for row in self.rows:
                m = Messages(
                    llm=row.llm.value,
                    language=row.language,
                )
                if row.system:
                    m.messages.append(SystemMessage(content=row.system))
                if row.input:
                    m.messages.append(UserMessage(content=row.input))
                if row.output and len(m) >= 1:
                    m.messages.append(AssistantMessage(content=row.output))
                if len(m) >= 2:
                    m_list.messages_list.append(m)
            return m_list
        raise NotImplementedError(f"Unsupported format: {fmt}")
    
    def to_file(self, file: str | Path = "all.json", dir: str | Path = DEFAULT_DATASET_DIR):
        print(f"Dumped {len(self.rows)} rows to {(Path(dir) / file).absolute().as_posix()!r}")
        (Path(dir) / file).write_text(self.model_dump_json())

    @classmethod
    def from_file(
        cls, 
        file: str | Path = "all.json", 
        dir: str | Path = DEFAULT_DATASET_DIR, 
        full_path: str | Path = None,
        log_errors = True,
    ):
        full_path = Path(full_path) if full_path else Path(dir) / file
        try:
            m = cls.model_validate_json(full_path.read_text())
            language = get_language(full_path)
            default_lang = DatasetRow.model_fields['language'].default
            if language is not None:
                for row in m.rows:
                    if row.language == default_lang and language != default_lang:
                        row.language = language
            return m
        except Exception as e:
            print(f"Got error: {e}") if log_errors else None
    
    @classmethod
    def from_files(
        cls, 
        files: list[str | Path] = [],
        dir: str | Path = None,
        log_errors = True,
        threads = 4,
    ):
        from_file_fp = partial(cls.from_file, None, None, log_errors=log_errors)
        if dir is None:
            data: list[tuple[str | Path, LLMDatasetBase]] = run_parallel_exec(
                from_file_fp, set(files), quiet=not log_errors, max_workers=threads
            )
        else:
            data: list[tuple[str | Path, LLMDatasetBase]] = run_parallel_exec(
                cls.from_file, set(files), dir, None, log_errors, quiet=not log_errors, max_workers=threads
            )
        data = [x[1].rows for x in data if isinstance(x[1], cls)]
        return cls(rows=list(chain(*data)))
    
    def fill_systems(self, systems: list[str] = None):
        _systems = {x.system for x in self.rows if x.system}
        if not isinstance(systems, list):
            systems = []
        systems = systems + list(_systems)
        systems = list({x for x in systems if isinstance(x, str)})
        if len(systems) == 0:
            return self
        for row in self.rows:
            if not row.system:
                row.system = random.choice(systems)
        return self.__class__(rows=self.rows)
    
    def unique(self):
        return self.__class__(
            rows=list({hash_uuid(x): x for x in self.rows}.values())
        )
        
    @classmethod
    def from_messages(cls, messages: AlpacaMessagesList, llm_type: LLMType, strict: bool = False):
        vals = [
            {
                "llm": x.llm,
                "language": x.language,
                "system": x.system,
                "input": x.input,
                "output": x.output,
            }
            for x in messages
        ]
        if strict:
            rows = [
                llm_row_factory(llm_type).try_model_validate(val=x, none_on_fail=True)
                for x in vals
            ]
        else:
            rows = [DatasetRow(**x) for x in vals]
        return cls(
            rows=[
                x for x in rows if None not in [x, getattr(x, "input", None), getattr(x, "output", None)]
            ]
        )

    @classmethod
    def from_dataset(cls, d: Dataset, llm_type: LLMType, strict: bool = False):
        return cls.from_messages(AlpacaMessagesList.from_dataset(d), llm_type, strict)
    
    @classmethod
    def from_jsonl(cls, jsonl_file: str | Path, llm_type: LLMType = None, strict: bool = False):
        jsonl_file = Path(jsonl_file)
        assert jsonl_file.exists(), f"File {jsonl_file!r} does not exist"
        if llm_type is None:
            llm_type = LLMType.from_substr(jsonl_file.as_posix(), none_on_fail=False)
        d = cls.from_dataset(Dataset.from_json(jsonl_file.as_posix()), llm_type, strict)
        default_lang = DatasetRow.model_fields['language'].default
        language = get_language(jsonl_file.as_posix())
        for m in d.rows:
            if language not in [None, default_lang] and m.language == default_lang:
                m.language = language
        return d

    def page(self, page_size: int = 10, offset: int = 0):
        return self.__class__(rows=self.rows[offset:offset+page_size])
    
    def page_iterator(self, page_size: int = 10, offset: int = 0, use_tqdm=True):
        if use_tqdm:
            from tqdm import tqdm
        else:
            tqdm = lambda x: x
        for i in tqdm(range(offset, len(self.rows), page_size)):
            yield self.__class__(rows=self.rows[i:i+page_size])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rows={len(self.rows)})"
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, index):
        rows = self.rows[index]
        return self.__class__(rows=rows if isinstance(rows, list) else [rows])
    
    def __iter__(self):
        return iter(self.rows)
    
    def __contains__(self, item):
        return item in self.rows
    
    def __add__(self, other: "LLMDatasetBase"):
        if type(self) != type(other):
            raise NotImplementedError(f"Cannot add {type(self)} and {type(other)}")
        _dst_merged = self.__class__(rows=self.rows + other.rows)
        _dst = self.__class__(rows=[])
        for llm in LLMType:
            _dst.rows += _dst_merged.get_llm(llm).fill_systems().unique().rows
        return _dst

class LLMDatasetWithTypes(LLMDatasetBase):
    rows: list[LLM1DatasetRow|LLM2DatasetRow|LLM3DatasetRow] = []

class LLMDataset(LLMDatasetBase):
    rows: list[DatasetRow] = []

    def get_llm_type_rows(self, llm_type: LLMType = None, verbose: bool = False):
        if llm_type:
            dataset = self.get_llm(llm_type)
        else:
            dataset = self
        rows = [
            llm_row_factory(row.llm).try_model_validate(
                row.model_dump(), none_on_fail=True, verbose=verbose
            )
            for row in dataset.rows
        ]
        rows = [row for row in rows if row is not None]
        llmdt = LLMDatasetWithTypes(rows=[
            row for row in rows 
            if isinstance(row, (LLM1DatasetRow, LLM2DatasetRow, LLM3DatasetRow)) 
            and None not in [row.input, row.output]
        ])
        return llmdt
        
