from models.base import CustomBaseModel as BaseModel
from helpers.utils import hash_uuid

class Question(BaseModel):
    question: str
    
    def __hash__(self) -> int:
        return hash_uuid(self.question).int

class QuestionSplit(Question):
    tasks: list[str]
    can_i_answer: bool

    def __hash__(self) -> int:
        return hash_uuid(self.question).int
