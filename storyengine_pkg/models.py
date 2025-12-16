"""
데이터 모델 정의
"""
from typing import TypedDict, List, Dict, Optional


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
    immediate_reaction: str  # 선택 직후 보여줄 즉각 반응 (100-200자)


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

# ============================================
# NEW PYDANTIC MODELS FOR SEQUENTIAL GENERATION
# ============================================

from pydantic import BaseModel, Field

class StoryConfig(BaseModel):
    num_episodes: int = Field(alias="numEpisodes")
    max_depth: int = Field(alias="maxDepth")
    selected_gauge_ids: List[str] = Field(alias="selectedGaugeIds")

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase

class InitialAnalysis(BaseModel):
    summary: Optional[str] = None
    characters: List[Dict]

class EpisodeModel(BaseModel):
    # Java에서 "order" 또는 "episodeOrder"로 보낼 수 있음
    episode_order: int = Field(alias="order")
    title: str

    # Java에서 "startNode" (단일) 또는 "nodes" (배열)로 보낼 수 있음
    start_node: Optional[Dict] = Field(default=None, alias="startNode")
    nodes: Optional[List[Dict]] = Field(default=None)

    # 추가 필드 (Java EpisodeDto에 있는 필드들)
    description: Optional[str] = None
    theme: Optional[str] = None
    intro_text: Optional[str] = Field(default=None, alias="introText")
    endings: Optional[List[Dict]] = None

    class Config:
        populate_by_name = True  # snake_case와 camelCase 모두 허용

class GenerateNextEpisodeRequest(BaseModel):
    initial_analysis: InitialAnalysis = Field(alias="initialAnalysis")
    story_config: StoryConfig = Field(alias="storyConfig")
    novel_context: str = Field(alias="novelContext")
    current_episode_order: int = Field(alias="currentEpisodeOrder")
    previous_episode: Optional[EpisodeModel] = Field(alias="previousEpisode", default=None)

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase
