from typing import Callable, Mapping, Self

from pydantic import BaseModel

from teenyagent.model import Model
from teenyagent.schema import ToolCallResult
from teenyagent.system_prompt import SystemPromptBuilder
from teenyagent.tool import Tool


class Memory:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._messages: list[dict] = []

    @property
    def messages(self) -> list[dict]:
        return self._messages

    @messages.setter
    def messages(self, messages: list[dict]):
        self._messages = messages


class Agent:
    def __init__(
        self,
        name: str,
        model: Model,
        system_prompt: str | None = None,
        description: str | None = None,
        response_type: BaseModel | None = None,
        memory: bool = False,
    ):
        self.name = name
        self.model = model
        self.description = description
        self.tools: Mapping[str, Tool] = {}
        self.sub_agents: Mapping[str, Self] = {}
        self.response_type = response_type if response_type else str
        self.system_prompt_builder = SystemPromptBuilder(self.response_type, system_prompt=system_prompt)
        self._memory = Memory(name) if memory else None

    @property
    def memory(self) -> bool:
        return bool(self._memory)

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

    def run(self, prompt: str):
        # Add system prompt
        messages = [self.system_prompt_builder.system_prompt]
        if self._memory:
            # Add stored message history
            messages += self._memory.messages

        # Add current user prompt
        messages += [{'role': 'user', 'content': prompt}]

        resp = self._loop(messages)

        if self._memory:
            # Set the updated conversation history to memory without system prompt
            self._memory.messages = messages[1:]

        return resp

    def _loop(self, messages: list[dict]):
        while True:
            resp = self.model.prompt(messages, self.response_type)
            messages.append(resp.model_dump())

            if resp.content.exit:
                return (
                    resp.content.response if isinstance(self.response_type, str) else resp.content.response.model_dump()
                )

            if resp.content.tool_calls:
                for tool_call in resp.content.tool_calls:
                    tool_id = tool_call.tool_id
                    tool_args = tool_call.tool_args
                    tool_result = self.tools[tool_id](**tool_args)
                    messages.append(ToolCallResult(tool_id=tool_id, content=tool_result).model_dump())

            if resp.content.sub_agent_calls:
                for sub_agent_call in resp.content.sub_agent_calls:
                    sub_agent_id = sub_agent_call.agent_id
                    sub_agent_prompt = sub_agent_call.prompt
                    sub_agent = self.sub_agents[sub_agent_id]
                    sub_agent_resp = sub_agent.run(sub_agent_prompt)
                    messages.append(ToolCallResult(tool_call_id=sub_agent_id, content=sub_agent_resp).model_dump())

    def __repr__(self) -> str:
        return (
            f'Agent('
            f'model={self.model.name}, '
            f'tools={[tool_name for tool_name in self.tools]}, '
            f'sub_agents={[sub_agent_name for sub_agent_name in self.sub_agents]}, '
            f'memory={self.memory})'
        )
