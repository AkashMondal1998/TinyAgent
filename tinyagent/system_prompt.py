from typing import TYPE_CHECKING

import jinja2

from tinyagent.schema import Content

if TYPE_CHECKING:
    from tinyagent.agent import Agent
    from tinyagent.tool import Tool

DEFAULT_RESPONSE_PROMPT = """ Follow this schema for generating responses
Response schema: {{response_schema}}

Populate the response field to give simple response or when you have the final response
Populate the tool_calls field if you need to call any tools
Populate the sub_agent_calls field if you to call any sub agents
Use the exit field when you think you have gathered the information and is ready to respond.

Use the inlcuded tools and sub agents if any.
Do not try to use tools or sub agents that are not there.

Always respond using valid JSON that follows the above schema.
"""


TOOL_DEFINITION = """
-------------TOOL-------------
    Name: {{tool.name}}
    Desciption: {{tool.description}}
------------------------------
"""
SUB_AGENT_DEFINITION = """
---------------SUB AGENT------
    Name: {{agent.name}}
    Desciption: {{agent.description}}
------------------------------
"""


class SystemPromptBuilder:
    def __init__(self, system_prompt: str | None = None):
        if system_prompt:
            self.system_prompt = {'role': 'system', 'content': system_prompt}
        else:
            self.system_prompt = {'role': 'system', 'content': ''}

        self.system_prompt['content'] += '\n\n' + jinja2.Template(DEFAULT_RESPONSE_PROMPT).render(
            response_schema=Content.model_json_schema()
        )

        self.tool_template = jinja2.Template(TOOL_DEFINITION)
        self.sub_agent_template = jinja2.Template(SUB_AGENT_DEFINITION)
        self.response_schema = jinja2.Template(DEFAULT_RESPONSE_PROMPT)

    def add_tool(self, tool: Tool):
        self.system_prompt['content'] += '\n\n' + self.tool_template.render(tool=tool)

    def add_sub_agent(self, sub_agent: Agent):
        self.system_prompt['content'] += '\n\n' + self.sub_agent_template.render(agent=sub_agent)
