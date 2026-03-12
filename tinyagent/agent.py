from typing import Callable, Mapping, Self

from pydantic import BaseModel, ValidationError

from tinyagent.model import Model
from tinyagent.schema import ModelResponse, ToolCallResult
from tinyagent.system_prompt import SystemPromptBuilder
from tinyagent.tool import Tool
from tinyagent.memory import Memory


class Agent:
    def __init__(
        self,
        name: str,
        model: Model,
        system_prompt: str | None = None,
        description: str | None = None,
        response_type: BaseModel | None = None,
    ):
        self.name = name
        self.model = model
        self.description = description
        self.tools: Mapping[str, Tool] = {}
        self.sub_agents: Mapping[str, Self] = {}
        self.response_type = response_type
        self.system_prompt_builder = SystemPromptBuilder(system_prompt=system_prompt, response_type=self.response_type)
        self.memory = Memory(name)

    def run(self, prompt: str):

        messages = (
            [self.system_prompt_builder.system_prompt] + self.memory.messages + [{'role': 'user', 'content': prompt}]
        )

        resp, new_messages = self._loop(messages)

        self.memory.save(new_messages)
        self.memory.save_to_database(new_messages)

        return resp

    def _loop(self, messages: list[dict]):
        while True:
            new_messages = []

            resp: ModelResponse = self._get_valid_response(messages)
            new_messages.append(messages[-1])
            new_messages.append(resp.model_dump())

            if resp.content.exit:
                return resp.content.response, new_messages

            if resp.content.tool_calls:
                for tool_call in resp.content.tool_calls:
                    tool_id = tool_call.tool_id
                    tool_args = tool_call.tool_args
                    tool_result = self.tools[tool_id](**tool_args)
                    new_messages.append(ToolCallResult(tool_id=tool_id, content=tool_result).model_dump())

            if resp.content.sub_agent_calls:
                for sub_agent_call in resp.content.sub_agent_calls:
                    sub_agent_id = sub_agent_call.agent_id
                    sub_agent_prompt = sub_agent_call.prompt
                    sub_agent = self.sub_agents[sub_agent_id]
                    sub_agent_resp = sub_agent.run(sub_agent_prompt)
                    new_messages.append(ToolCallResult(tool_call_id=sub_agent_id, content=sub_agent_resp).model_dump())

    def _get_valid_response(self, messages):
        tries = 3
        while tries > 0:
            resp: ModelResponse = self.model.prompt(messages)
            if not self.response_type:
                return resp
            if self._validate_response_format(resp.content.response):
                return resp
            tries -= 1
        raise ValueError('Unable to provide structured output')

    def _validate_response_format(self, response):
        try:
            self.response_type.model_validate(response)
            return True
        except ValidationError:
            return False

    def add_tool(self, func: Callable) -> None:
        tool = Tool(func.__name__, func.__doc__, func)
        self.tools[func.__name__] = tool
        self.system_prompt_builder.add_tool(tool)

    def tool(self, func: Callable) -> Callable:
        self.add_tool(func)
        return func

    def add_sub_agent(self, agent: Self) -> None:
        self.sub_agents[agent.name] = agent
        self.system_prompt_builder.add_sub_agent(agent)

    def __repr__(self) -> str:
        return f'Agent(model={self.model.name} tools={[tool for tool in self.tools.values()]})'
