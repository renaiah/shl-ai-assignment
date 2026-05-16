from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from agent import get_response

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = get_response(messages)
    return ChatResponse(
        reply=result["reply"],
        recommendations=result.get("recommendations", []),
        end_of_conversation=result.get("end_of_conversation", False)
    )