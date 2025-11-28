"""
데이터 모델 정의
"""
from typing import TypedDict, List, Dict


class Character(TypedDict):
    name: str
    aliases: List[str]
    description: str
    relationships: List[str]


class Gauge(TypedDict):
    id: str
    name: str
    meaning: str
    min_label: str
    max_label: str
    description: str
    initial_value: int  # AI가 소설 상황에 맞게 설정 (0~100)


class FinalEnding(TypedDict):
    id: str
    type: str
    title: str
    condition: str
    summary: str


class EpisodeEnding(TypedDict):
    id: str
    title: str
    condition: str
    text: str
    gauge_changes: Dict[str, int]


class StoryNodeDetail(TypedDict):
    npc_emotions: Dict[str, str]
    situation: str
    relations_update: Dict[str, str]


class StoryChoice(TypedDict):
    text: str
    tags: List[str]


class StoryNode(TypedDict):
    id: str
    depth: int
    text: str
    details: StoryNodeDetail
    choices: List[StoryChoice]
    parent_id: str | None
    node_type: str
    episode_id: str


class Episode(TypedDict):
    id: str
    title: str
    order: int
    description: str
    theme: str
    intro_text: str
    nodes: List[StoryNode]
    endings: List[EpisodeEnding]
