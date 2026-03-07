import jinja2
from pydantic import BaseModel

from tinyagent.schema import Prompt

SYSTEM_PROMPT = """ Follow this schema for generating responses
Response schema: {{response_schema}}

Populate the response field to give simple response or when you have the final response
Populate the tool_calls field if you need to call any tools
Populate the sub_agent_calls field if you to call any sub agents
Use the exit field when you think you have gathered the information and is ready to respond.
"""

TOOL_DEFINATIONS = """You have these tools at your disposal
---------------TOOLS---------------
{% for tool_name,tool_metadata in tools.items() %}
    Name: {{tool_name}}
    Desciption:
    {{tool_metadata.description}}
{% endfor %}

Choose the appropiate tool to use


"""


SUB_AGENTS_DEFINATIONS = """You have these agents at your disposal
---------------AGENTS---------------
{% for agent_name,agent in agents.items() %}
    Name: {{agent_name}}
    Desciption:
    {{agent.description}}
{% endfor %}

Choose the appropiate agent to use
"""


def generate_system_prompt(
    system_prompt: Prompt | None = None,
    response_type: BaseModel | None = None,
    tools: dict | None = None,
    sub_agents: dict | None = None,
) -> Prompt:
    if not system_prompt:
        system_prompt = Prompt(role='system', content='')

    if response_type:
        system_prompt.content += '\n' + jinja2.Template(SYSTEM_PROMPT).render(
            response_schema=response_type.model_json_schema()
        )
    if tools:
        system_prompt.content += '\n' + jinja2.Template(TOOL_DEFINATIONS).render(tools=tools)
    if sub_agents:
        system_prompt.content += '\n' + jinja2.Template(SUB_AGENTS_DEFINATIONS).render(agents=sub_agents)

    return system_prompt
