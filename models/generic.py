from pydantic import BaseModel

class Question(BaseModel):
    question: str

class QuestionSplit(Question):
    tasks: list[str]
    can_i_answer: bool
