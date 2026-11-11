"""Prompts live in git (config/prompts/*.yaml). Each has a name and a version, and the
`name@version` label is recorded with every answer so results can be traced to a prompt."""

from dataclasses import dataclass
from functools import lru_cache

import yaml

from rag.core.config import PROMPTS_DIR


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str
    user: str

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    def render_user(self, **variables: str) -> str:
        return self.user.format(**variables)


@lru_cache
def load_prompt(name: str) -> Prompt:
    data = yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text())
    return Prompt(name=data["name"], version=data["version"], system=data["system"], user=data["user"])
