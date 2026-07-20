# ============================================================
#  api/schemas.py  –  Pydantic request / response models
# ============================================================

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    user_id:  str
    question: str


class DeleteRequest(BaseModel):
    user_id: str


class DeletePDFRequest(BaseModel):
    user_id:  str
    filename: str


class AnswerResponse(BaseModel):
    user_id:  str
    question: str
    answer:   str
    scores:   dict


class PDFListResponse(BaseModel):
    user_id: str
    files:   list[str]


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: str
