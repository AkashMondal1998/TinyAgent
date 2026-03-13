import json
from typing import Literal,Any

from pydantic import BaseModel, Field, Json, field_serializer, field_validator


class ToolCall(BaseModel):
    tool_id: str = Field(description='Id of the tool')
    tool_args: dict = Field(description='Mapping of tool arg name and tool arg value')


class SubAgentCall(BaseModel):
    agent_id: str = Field(description='Id of the agent')
    prompt: str = Field(description='Prompt for the sub agent')


class Content(BaseModel):
    response: Any | None = Field(default=None, description='Response')
    tool_calls: list[ToolCall] = Field(default=[], description='list of tool calls')
    sub_agent_calls: list[SubAgentCall] = Field(default=[], description='list of sub agent calls')
    exit: bool = Field(default=False, description='Continue evaluation or not')


class ModelResponse(BaseModel):
    role: Literal['assistant']
    content: Json[Content]
    thinking: str | None = None

    @field_serializer('content', mode='plain')
    def ser(self, value: Content) -> str:
        return value.model_dump_json()


class ToolCallResult(BaseModel):
    role: Literal['tool'] = 'tool'
    content: str
    tool_call_id: str | None = None

    @field_validator('content', mode='before')
    @classmethod
    def validate_content(cls, value):
        if not isinstance(value, str):
            value = json.dumps(value)
        return value
