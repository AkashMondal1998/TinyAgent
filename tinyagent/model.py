import os

import requests
from pydantic import BaseModel

from tinyagent.schema import Message


class Model:
    def __init__(self, name: str):
        self.name = name
        self.api_key = os.getenv('OLLAMA_API_KEY')

    def prompt(self, messages: list[dict], response_type: BaseModel | str) -> Message[BaseModel | str]:
        data = {
            'model': self.name,
            'messages': messages,
            'stream': False,
            'format': 'json',
            'options': {'temperature': 0},
        }
        with requests.post(
            'https://ollama.com/api/chat', json=data, headers={'Authorization': f'Bearer {self.api_key}'}
        ) as response:
            model_response = Message[response_type](**response.json()['message'])
            return model_response
