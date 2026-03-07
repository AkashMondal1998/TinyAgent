from typing import Callable


class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def __repr__(self):
        return f'Tool(callable={self.name})'

    def __call__(self, **kwargs):
        return self.func(**kwargs)
