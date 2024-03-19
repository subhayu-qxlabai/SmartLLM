from models.base import CustomBaseModel as BaseModel


class Question(BaseModel):
    question: str
    
    def __hash__(self) -> int:
        return hash(self.question)

class QuestionSplit(Question):
    tasks: list[str]
    can_i_answer: bool

    def __hash__(self) -> int:
        return hash(self.question)
