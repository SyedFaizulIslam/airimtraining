from pydantic import BaseModel

class ChatPrompt(BaseModel):
    prompt: str
    systemrole: str