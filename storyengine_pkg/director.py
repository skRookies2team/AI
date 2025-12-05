import json
import operator
import re
import uuid
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from storyengine_pkg.models import (
    Character,
    Gauge,
    FinalEnding,
    EpisodeEnding,
    Episode,
    StoryNode,
    StoryChoice,
    StoryNodeDetail,
)

# ==============================================================================
# 2. 메인 클래스: 인터랙티브 스토리 디렉터
# ==============================================================================

class InteractiveStoryDirector:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        self.json_parser = JsonOutputParser()

    # --------------------------------------------------------------------------
    # [2단계] 등장인물 자동 추출 (Extract Characters)
    # --------------------------------------------------------------------------
    async def extract_characters(self, novel_text: str) -> List[Character]:
        print("🕵️ 등장인물 분석 중...")

        # 텍스트에 줄 번호 추가 (cite 참조용)
        lines = novel_text.split('\n')

        # 전체 텍스트 사용 (너무 길면 앞/중간/뒤 샘플링)
        if len(lines) <= 1000:
            selected_lines = lines
        else:
            # 앞 400줄, 중간 200줄, 뒤 400줄
            mid_start = len(lines) // 2 - 100
            selected_lines = lines[:400] + lines[mid_start:mid_start+200] + lines[-400:]

        numbered_text = '\n'.join([f"[{i+1}] {line}" for i, line in enumerate(selected_lines)])

        prompt = f"""당신은 문학 분석 전문가입니다. 아래 소설 텍스트에서 주요 등장인물들의 정보를 상세히 추출하세요.

[소설 텍스트] (줄 번호 포함)
{numbered_text}

[추출 항목]
각 캐릭터에 대해 다음 정보를 추출하세요:

1. **name**: 이름
2. **aliases**: 별명 리스트 (텍스트에서 불리는 다른 호칭들)
3. **description**: 통합 설명 - 다음을 모두 포함하여 상세히 작성:
   - 외형 묘사 (나이, 신체적 특징)
   - 성격 특성
   - 주요 행동 및 사건
   - 캐릭터의 변화/성장
   - 각 정보의 근거를 [cite: 줄번호] 형식으로 표기

4. **relationships**: 다른 인물과의 관계 리스트
   - 각 관계를 구체적인 사건/장면과 함께 설명
   - [cite: 줄번호] 형식으로 근거 표기

[예시]
{{
    "name": "랠프",
    "aliases": ["금발의 소년", "대장"],
    "description": "금발의 소년으로 [cite: 8, 35], 만 12살입니다 [cite: 60]. 딱 벌어진 어깨와 부드러운 눈가를 지녔습니다 [cite: 61]. 수영을 잘하며 [cite: 90] 소라를 불어 아이들을 소집했습니다 [cite: 131]. 투표를 통해 대장으로 선출되었으며 [cite: 224], 봉화를 피워 구조되는 것을 목표로 삼습니다 [cite: 436]. 사이먼의 죽음에 죄책감을 느꼈고 [cite: 2057], 구조 직후 울음을 터뜨립니다 [cite: 2687].",
    "relationships": [
        "새끼돼지와 처음 만나 함께 행동함 [cite: 8-18]",
        "잭과 리더십 문제로 대립함 [cite: 612, 892]",
        "새끼돼지의 별명을 폭로했으나 나중에는 신뢰함 [cite: 215, 1840]"
    ]
}}

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "characters": [
        {{
            "name": "캐릭터 이름",
            "aliases": ["별명1", "별명2"],
            "description": "상세 설명 [cite: 줄번호]...",
            "relationships": ["관계 설명 [cite: 줄번호]", ...]
        }}
    ]
}}"""
        response = await self.llm.ainvoke(prompt)
        characters = self._parse_json(response.content).get("characters", [])

        # 빈 결과일 경우 기본값 반환
        if not characters:
            print("  ⚠️ 캐릭터 추출 실패, 기본값 사용")
            return [
                {
                    "name": "주인공",
                    "aliases": [],
                    "description": "주인공에 대한 정보를 추출할 수 없습니다.",
                    "relationships": []
                }
            ]

        return characters

    # --------------------------------------------------------------------------
    # [3단계] 게이지 제안 (Generate Gauges)
    # --------------------------------------------------------------------------
    async def suggest_gauges(self, novel_summary: str) -> List[Gauge]:
        print("📊 스토리 게이지 설계 중...")
        prompt = f"""이 소설의 핵심 갈등과 테마를 관통하는 '게이지(수치)' 시스템을 설계하려 합니다.
가장 적절한 5개의 게이지를 제안해주세요.

[소설 요약]
{novel_summary}

[요구사항]
각 게이지에 대해 다음을 정의하세요:
- id: 영문 소문자 식별자 (예: "civilization", "fear")
- name: 게이지 한글 이름 (예: "문명도", "공포심")
- meaning: 이 게이지가 의미하는 바 (1-2문장)
- min_label: 0일 때의 상태 (예: "야만", "평온")
- max_label: 100일 때의 상태 (예: "질서", "공포")
- description: 스토리에서 이 게이지가 어떻게 사용되는지 설명
- initial_value: 소설 시작 시점의 초기값 (0~100, 소설 상황에 맞게 설정)
  - 예: 평화로운 시작이면 hope=70, 위기 상황이면 hope=30

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "gauges": [
        {{
            "id": "civilization",
            "name": "문명도",
            "meaning": "사회 질서와 규범을 유지하려는 정도",
            "min_label": "야만",
            "max_label": "질서",
            "description": "높을수록 민주적 리더십과 규칙을 따르고, 낮을수록 본능과 폭력에 의존",
            "initial_value": 65
        }}
    ]
}}"""
        response = await self.llm.ainvoke(prompt)
        gauges = self._parse_json(response.content).get("gauges", [])

        # 빈 결과일 경우 기본값 반환
        if not gauges:
            print("  ⚠️ 게이지 제안 실패, 기본값 사용")
            return [
                {
                    "id": "progress",
                    "name": "진행도",
                    "meaning": "스토리 진행 상황",
                    "min_label": "시작",
                    "max_label": "완료",
                    "description": "스토리가 얼마나 진행되었는지 나타냄"
                }
            ]

        return gauges

    # --------------------------------------------------------------------------
    # [4단계] 최종 엔딩 생성 (Generate Final Endings - 게이지 누적 기반)
    # --------------------------------------------------------------------------
    async def design_final_endings(
        self,
        novel_summary: str,
        selected_gauges: List[Gauge],
        ending_config: Dict[str, int] = None
    ) -> List[FinalEnding]:
        """
        최종 엔딩 설계

        Args:
            ending_config: 엔딩 타입별 개수 설정
                예: {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}
                지원 타입: happy, tragic, neutral, open, bad, bittersweet
        """
        # 기본값 설정
        if ending_config is None:
            ending_config = {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}

        total_endings = sum(ending_config.values())
        print(f"🏁 최종 엔딩 설계 중 ({total_endings}개)...")

        # 게이지 정보 포맷팅
        gauges_detail = []
        for g in selected_gauges:
            gauge_str = f"• {g.get('name', '?')} ({g.get('id', '?')}): {g.get('min_label', '?')} (0) ↔ {g.get('max_label', '?')} (100)"
            gauges_detail.append(gauge_str)
        gauges_info = "\n".join(gauges_detail)

        # 엔딩 타입 요구사항 생성
        ending_requirements = []
        type_descriptions = {
            "happy": "행복한 엔딩 (희망적인 결말, 목표 달성)",
            "tragic": "비극적인 엔딩 (파멸, 죽음, 실패)",
            "neutral": "중립적인 엔딩 (무난한 결말, 큰 변화 없음)",
            "open": "열린 결말 (해석의 여지, 미완의 이야기)",
            "bad": "나쁜 엔딩 (불행한 결말, 손실)",
            "bittersweet": "씁쓸한 엔딩 (희생을 통한 성공, 달콤쓴 결말)"
        }

        for ending_type, count in ending_config.items():
            if count > 0:
                desc = type_descriptions.get(ending_type, ending_type)
                ending_requirements.append(f"- {desc}: {count}개")

        ending_requirements_str = "\n".join(ending_requirements)

        prompt = f"""선택된 게이지의 최종 누적 수치에 따라 도달할 수 있는 최종 엔딩을 설계하세요.

[소설 요약]
{novel_summary}

[게이지 시스템]
{gauges_info}

[엔딩 타입별 요구사항]
다음 타입과 개수에 맞춰 엔딩을 생성해주세요:
{ending_requirements_str}

총 {total_endings}개의 엔딩을 생성해야 합니다. 

[중요]
- 이 엔딩들은 여러 에피소드를 거친 후 누적된 게이지 값으로 결정됩니다.
- 각 에피소드 엔딩에서 게이지가 +/- 되어 최종 값이 결정됩니다.
- 게이지 초기값은 50이며, 0~100 범위입니다.

[요구사항]
각 엔딩에 대해 다음을 정의하세요:
- id: 영문 소문자 식별자 (예: "ending_hope", "ending_despair")
- type: 엔딩 타입 (예: "happy", "bad", "neutral", "tragic", "open")
- title: 엔딩 제목 (예: "구조의 희망", "야만으로의 추락")
- condition: 도달 조건 - 최종 게이지 상태 (예: "hope >= 70 AND despair <= 30")
- summary: 엔딩 내용 요약 (3-5문장)

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "endings": [
        {{
            "id": "ending_hope",
            "type": "happy",
            "title": "구조의 희망",
            "condition": "hope >= 70 AND trust >= 60",
            "summary": "소년들은 끝까지 희망을 잃지 않고 서로를 신뢰했다. 마침내 구조선이 도착하고, 모두가 안전하게 돌아간다."
        }},
        {{
            "id": "ending_despair",
            "type": "tragic",
            "title": "절망의 나락",
            "condition": "despair >= 70 AND trust <= 30",
            "summary": "절망이 모든 것을 집어삼켰다. 서로에 대한 신뢰는 무너지고, 비극적인 결말을 맞이한다."
        }}
    ]
}}"""
        response = await self.llm.ainvoke(prompt)
        endings = self._parse_json(response.content).get("endings", [])

        # 빈 결과일 경우 기본값 반환
        if not endings:
            print("  ⚠️ 최종 엔딩 설계 실패, 기본값 사용")
            return [
                {
                    "id": "ending_default",
                    "type": "neutral",
                    "title": "기본 엔딩",
                    "condition": "default",
                    "summary": "스토리가 기본적인 결말에 도달합니다."
                }
            ]

        print(f"  ✅ {len(endings)}개의 최종 엔딩 설계 완료")
        return endings

    # --------------------------------------------------------------------------
    # [5단계] 에피소드 분할 (Split into Episodes)
    # --------------------------------------------------------------------------
    async def split_into_episodes(self, novel_summary: str, characters: List[Character], num_episodes: int = 4) -> List[Dict]:
        print(f"📚 소설을 {num_episodes}개 에피소드로 분할 중...")

        # 캐릭터 이름 목록
        char_names = [c.get('name', '이름없음') for c in characters]

        prompt = f"""주어진 소설을 {num_episodes}개의 독립적인 에피소드로 분할하세요.

[소설 요약]
{novel_summary}

[등장인물]
{', '.join(char_names)}

[중요 규칙]
- 각 에피소드는 스토리상 서로 연결되지 않습니다 (독립적)
- 에피소드 간에는 오직 게이지만 누적됩니다
- 각 에피소드는 자체적인 시작, 전개, 엔딩을 가집니다

[요구사항]
각 에피소드에 대해 다음을 정의하세요:
- id: 영문 소문자 식별자 (예: "ep1_encounter")
- title: 에피소드 제목
- order: 순서 (1, 2, 3...)
- description: 에피소드 요약 (2-3문장)
- theme: 핵심 테마/갈등 (예: "신뢰 vs 의심", "희망 vs 절망")
- key_characters: 주요 등장인물 리스트

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "episodes": [
        {{
            "id": "ep1_encounter",
            "title": "첫 만남",
            "order": 1,
            "description": "주인공들이 처음 만나 서로를 알아가는 과정. 첫인상과 초기 관계가 형성된다.",
            "theme": "신뢰 형성",
            "key_characters": ["랠프", "잭", "새끼돼지"]
        }}
    ]
}}"""
        response = await self.llm.ainvoke(prompt)
        episodes = self._parse_json(response.content).get("episodes", [])

        if not episodes:
            print("  ⚠️ 에피소드 분할 실패, 기본값 사용")
            return [
                {
                    "id": "ep1_default",
                    "title": "에피소드 1",
                    "order": 1,
                    "description": "기본 에피소드",
                    "theme": "기본",
                    "key_characters": char_names[:3]
                }
            ]

        print(f"  ✅ {len(episodes)}개 에피소드 분할 완료")
        for ep in episodes:
            print(f"    • [{ep.get('order', '?')}] {ep.get('title', '제목없음')}: {ep.get('theme', '')}")

        return episodes

    # --------------------------------------------------------------------------
    # [5-2단계] 에피소드 도입부 생성 (Generate Episode Intro)
    # --------------------------------------------------------------------------
    async def generate_episode_intro(self, episode: Dict, characters: List[Character], novel_summary: str) -> str:
        print(f"  🎬 '{episode.get('title', '?')}' 도입부 생성 중...")

        # 캐릭터 정보
        char_names = [c.get('name', '이름없음') for c in characters]

        prompt = f"""다음 에피소드의 도입부를 작성해주세요.
플레이어가 첫 번째 선택지를 만나기 전에 읽게 되는 스토리입니다.

[소설 배경]
{novel_summary}

[에피소드 정보]
- 제목: {episode.get('title', '?')}
- 설명: {episode.get('description', '?')}
- 테마: {episode.get('theme', '?')}
- 주요 등장인물: {', '.join(episode.get('key_characters', char_names[:3]))}

[작성 요구사항]
1. **분량**: 800~1200자
2. **내용 포함**:
   - 현재 상황과 배경 설명
   - 등장인물들의 대화와 행동
   - 분위기와 감정 묘사
   - 에피소드의 핵심 갈등/테마 암시
3. **스타일**:
   - 소설처럼 생생하게 묘사
   - 플레이어가 상황에 몰입할 수 있도록
   - 마지막에 자연스럽게 선택의 순간으로 이어지도록

도입부 텍스트만 작성해주세요 (JSON 아님):"""

        response = await self.llm.ainvoke(prompt)
        intro_text = response.content.strip()

        print(f"    ✅ 도입부 생성 완료 ({len(intro_text)}자)")
        return intro_text

    # --------------------------------------------------------------------------
    # [6단계] 에피소드 엔딩 설계 (Design Episode Endings)
    # --------------------------------------------------------------------------
    async def design_episode_endings(self, episode: Dict, selected_gauges: List[Gauge], num_endings: int = 3) -> List[EpisodeEnding]:
        print(f"  🎯 '{episode.get('title', '?')}' 에피소드 엔딩 설계 중...")

        # 게이지 정보 포맷팅
        gauges_info = self._format_gauges(selected_gauges)

        prompt = f"""이 에피소드의 {num_endings}가지 엔딩을 설계하세요. 각 엔딩은 플레이어의 선택 태그 누적에 따라 도달하며, 게이지에 영향을 줍니다.

[에피소드 정보]
- 제목: {episode.get('title', '?')}
- 설명: {episode.get('description', '?')}
- 테마: {episode.get('theme', '?')}

[게이지 시스템]
{gauges_info}

[선택지 태그 시스템]
플레이어가 선택지를 고를 때마다 해당 태그가 누적됩니다.
사용 가능한 태그: cooperative, aggressive, cautious, trusting, doubtful, brave, fearful, rational, emotional

[중요]
- 각 엔딩에서만 게이지가 변화합니다
- 게이지 변화량은 엔딩의 중요도와 극적 효과에 따라 자유롭게 설정하세요:
  - 작은 영향: -10 ~ +10
  - 보통 영향: -20 ~ +20
  - 큰 영향 (극적인 엔딩): -30 ~ +30
- condition은 태그 점수 기반 조건식으로 작성 (예: "cooperative >= 2", "trusting > doubtful")

[요구사항]
각 엔딩에 대해 다음을 정의하세요:
- id: 영문 소문자 식별자
- title: 엔딩 제목
- condition: 태그 기반 조건식 (예: "cooperative >= 2 AND trusting >= 1")
- text: 엔딩 텍스트 (3-5문장)
- gauge_changes: 게이지 변화 (예: {{'hope': 15, 'trust': 10}})

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "endings": [
        {{
            "id": "ep1_ending_trust",
            "title": "신뢰의 시작",
            "condition": "cooperative >= 2 AND trusting >= 1",
            "text": "서로를 알아가며 신뢰가 싹텄다. 아직 완전하지는 않지만, 함께할 수 있다는 희망이 생겼다.",
            "gauge_changes": {{"hope": 10, "trust": 15}}
        }},
        {{
            "id": "ep1_ending_doubt",
            "title": "의심의 씨앗",
            "condition": "doubtful >= 2 OR aggressive >= 2",
            "text": "서로를 경계하며 거리를 두었다. 불신의 씨앗이 마음 속에 심어졌다.",
            "gauge_changes": {{"hope": -5, "trust": -10}}
        }},
        {{
            "id": "ep1_ending_neutral",
            "title": "조심스러운 관망",
            "condition": "default",
            "text": "특별한 진전 없이 에피소드가 마무리되었다. 아직 서로에 대해 알아가는 중이다.",
            "gauge_changes": {{"hope": 0, "trust": 0}}
        }}
    ]
}}"""
        response = await self.llm.ainvoke(prompt)
        endings = self._parse_json(response.content).get("endings", [])

        if not endings:
            print(f"    ⚠️ 에피소드 엔딩 설계 실패, 기본값 사용")
            return [
                {
                    "id": f"{episode.get('id', 'ep')}_ending_default",
                    "title": "기본 엔딩",
                    "condition": "기본",
                    "text": "에피소드가 끝났습니다.",
                    "gauge_changes": {{}}
                }
            ]

        print(f"    ✅ {len(endings)}개 엔딩 설계 완료")
        return endings

    # --------------------------------------------------------------------------
    # [5단계] 스토리 트리 생성 (Generate Story Tree - LangGraph Engine)
    # --------------------------------------------------------------------------
    async def generate_full_tree(self, context: Dict, max_depth: int = 3) -> List[StoryNode]:
        """
        LangGraph를 사용하여 전체 스토리 트리를 생성합니다.

        Args:
            context: 캐릭터, 게이지, 엔딩, 구조가이드, 소설요약 등 모든 컨텍스트 정보
            max_depth: 트리의 최대 깊이 (기본값: 3)

        Returns:
            생성된 모든 StoryNode 리스트

        Note:
            - 깊이 0: 루트 노드 (1개)
            - 깊이 1: 루트의 선택지 수만큼 (예: 3개)
            - 깊이 2: 깊이1 노드들의 선택지 총합 (예: 9개)
            - ...
            - 기하급수적으로 증가하므로 max_depth를 적절히 설정해야 함
        """
        print(f"🌳 스토리 트리 생성 엔진 가동... (최대 깊이: {max_depth})")

        # 예상 노드 수 계산 (선택지가 평균 3개라 가정)
        avg_choices = 3
        estimated_nodes = sum([avg_choices ** d for d in range(max_depth + 1)])
        print(f"  📊 예상 노드 수: 약 {estimated_nodes}개")

        # LangGraph 워크플로우 구성
        workflow = StateGraph(StoryGenerationState)
        workflow.add_node("generate_node", self._node_generator)
        workflow.add_edge(START, "generate_node")
        workflow.add_conditional_edges("generate_node", self._plan_next_step)

        app = workflow.compile()

        # 초기 게이지 상태 설정 (AI가 제안한 initial_value 사용, 없으면 50)
        initial_gauges = {}
        for gauge in context.get("gauges", []):
            gauge_id = gauge.get("id", gauge.get("name", "unknown"))
            initial_gauges[gauge_id] = gauge.get("initial_value", 50)

        # 초기 상태 주입
        initial_state: StoryGenerationState = {
            "nodes": [],
            "context": context,
            "max_depth": max_depth,
            "current_gauges": initial_gauges
        }

        try:
            final_state = await app.ainvoke(initial_state)
            nodes = final_state.get("nodes", [])

            print(f"✅ 트리 생성 완료! 총 {len(nodes)}개 노드 생성됨")

            # 트리 구조 요약 출력
            self._print_tree_summary(nodes)

            return nodes

        except Exception as e:
            print(f"❌ 트리 생성 중 오류 발생: {e}")
            raise

    def _print_tree_summary(self, nodes: List[StoryNode]):
        """생성된 트리 구조 요약 출력"""
        if not nodes:
            return

        depth_counts = {}
        for node in nodes:
            depth = node.get("depth", 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        print("\n📈 트리 구조 요약:")
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            indent = "  " * depth
            print(f"  {indent}깊이 {depth}: {count}개 노드")

    # --- LangGraph 내부 로직 (Worker) ---
    async def _node_generator(self, state: Dict):
        """실제 LLM을 호출하여 스토리 노드를 생성하는 워커"""

        # Send로 전달된 task 정보 또는 초기 상태에서 추출
        if "task" in state:
            task = state["task"]
            depth = task["depth"]
            parent = task.get("parent_node")
            choice_taken = task.get("choice_taken")
            context = state["context"]
        else:
            # 초기 루트 노드 생성
            depth = 0
            parent = None
            choice_taken = None
            context = state["context"]

        # 노드 타입 결정 (AI가 선택지 개수는 자동 판단)
        max_depth = state.get("max_depth", 5)
        if depth == max_depth:
            node_type = "ending"
        elif depth == 0:
            node_type = "first_choice"
        elif depth == max_depth - 1:
            node_type = "climax"
        else:
            node_type = "development"

        # 캐릭터 정보 포맷팅
        characters_info = self._format_characters(context.get("characters", []))

        # 게이지 정보 포맷팅
        gauges_info = self._format_gauges(context.get("gauges", []))

        # 엔딩 정보 포맷팅
        endings_info = self._format_endings(context.get("endings", []))

        # 이전 스토리 컨텍스트 구성
        previous_context = ""
        if parent:
            previous_context = f"\n[이전 스토리]\n{parent.get('text', '')}\n\n[플레이어의 선택]\n{choice_taken.get('text', '') if choice_taken else '(시작)'}"

        # 현재 게이지 상태 계산
        current_gauges = self._calculate_current_gauges(state, choice_taken)

        system_prompt = f"""당신은 인터랙티브 소설 작가입니다. 주어진 컨텍스트를 바탕으로 스토리 노드를 생성합니다.

[소설 배경]
{context.get('novel_summary', '정보 없음')}

[등장인물]
{characters_info}

[게이지 시스템]
{gauges_info}

[현재 게이지 상태]
{json.dumps(current_gauges, ensure_ascii=False)}

[가능한 엔딩들]
{endings_info}

[현재 노드 정보]
- 깊이: {depth}/{max_depth}
- 노드 타입: {node_type}
- 선택지 개수: 상황에 맞게 2~4개 중 자동 결정

{previous_context}"""

        user_prompt = f"""위 컨텍스트를 바탕으로 다음 스토리 노드를 생성하세요.

[작성 요구사항]
1. **스토리 본문** (500-800자): 현재 상황을 생생하게 묘사. 캐릭터들의 대화와 행동 포함.
2. **디테일 정보**:
   - npc_emotions: 현재 등장하는 NPC들의 감정 상태 (예: {{'랠프': '불안', '잭': '흥분'}})
   - situation: 현재 상황 한 줄 요약
   - relations_update: 이번 장면으로 인한 인물 관계 변화 (예: {{'랠프-잭': '적대감 상승'}})
3. **선택지** (2~4개, 상황에 맞게 판단):
   - 선택지 개수는 현재 상황의 복잡도와 중요도에 따라 2~4개 중 적절히 결정하세요
     - 단순한 상황, 긴박한 순간: 2개
     - 일반적인 상황: 3개
     - 중요한 분기점, 다양한 접근이 가능한 상황: 4개
   - 선택지 텍스트는 플레이어 관점에서 1인칭으로 작성
   - 각 선택지에 특성 태그 포함 (1~2개씩)
   - 사용 가능한 태그: cooperative, aggressive, cautious, trusting, doubtful, brave, fearful, rational, emotional

{"⚠️ 이것은 에피소드 엔딩으로 연결되는 노드입니다. 스토리를 적절히 마무리하고 선택지는 빈 배열로 두세요." if node_type == "ending" else ""}

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "text": "스토리 본문...",
    "details": {{
        "npc_emotions": {{"캐릭터명": "감정"}},
        "situation": "상황 요약",
        "relations_update": {{ "관계": "변화 내용" }}
    }},
    "choices": [
        {{
            "text": "선택지 텍스트",
            "tags": ["cooperative", "trusting"]
        }},
        {{
            "text": "다른 선택지",
            "tags": ["aggressive", "doubtful"]
        }}
    ]
}}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            parsed = self._parse_json(response.content)

            # 노드 ID 생성
            node_id = str(uuid.uuid4())[:8]

            # 노드 구성
            new_node: StoryNode = {
                "id": node_id,
                "depth": depth,
                "text": parsed.get("text", "스토리 생성 실패"),
                "details": parsed.get("details", {
                    "npc_emotions": {},
                    "situation": "알 수 없음",
                    "relations_update": {}
                }),
                "choices": parsed.get("choices", []),
                "parent_id": parent["id"] if parent else None,
                "node_type": node_type,
                "episode_id": context.get("episode_id", "unknown")
            }

            print(f"  ✅ 노드 생성 완료: depth={depth}, id={node_id}, choices={len(new_node['choices'])}")

            return {"nodes": [new_node], "current_gauges": current_gauges}

        except Exception as e:
            print(f"  ❌ 노드 생성 실패 (depth={depth}): {e}")
            # 폴백 노드 생성
            fallback_node: StoryNode = {
                "id": str(uuid.uuid4())[:8],
                "depth": depth,
                "text": f"[오류로 인해 스토리를 생성할 수 없습니다: {str(e)}]",
                "details": {
                    "npc_emotions": {},
                    "situation": "오류 발생",
                    "relations_update": {}
                },
                "choices": [],
                "parent_id": parent["id"] if parent else None,
                "node_type": "error",
                "episode_id": context.get("episode_id", "unknown")
            }
            return {"nodes": [fallback_node], "current_gauges": current_gauges}

    def _format_characters(self, characters: List[Character]) -> str:
        """캐릭터 정보를 프롬프트용 문자열로 포맷팅"""
        if not characters:
            return "등록된 캐릭터 없음"

        result = []
        for char in characters:
            # cite 태그 제거하여 프롬프트에 사용 (간결하게)
            description = char.get('description', '정보 없음')
            # [cite: ...] 패턴 제거
            import re
            clean_desc = re.sub(r'\\\[cite:.*?\\\\]', '', description).strip()
            # 너무 길면 줄임
            if len(clean_desc) > 300:
                clean_desc = clean_desc[:300] + "..."

            char_info = f"""• {char.get('name', '이름없음')} (별명: {', '.join(char.get('aliases', []))})
  - 설명: {clean_desc}
  - 관계: {'; '.join(char.get('relationships', [])[:3])}"""
            result.append(char_info)

        return "\n".join(result)

    def _format_gauges(self, gauges: List[Gauge]) -> str:
        """게이지 정보를 프롬프트용 문자열로 포맷팅"""
        if not gauges:
            return "등록된 게이지 없음"

        result = []
        for g in gauges:
            gauge_info = f"""• {g.get('name', '이름없음')} (id: {g.get('id', 'unknown')})
  - 의미: {g.get('meaning', '불명')}
  - 0: {g.get('min_label', '최소')} ↔ 100: {g.get('max_label', '최대')}"""
            result.append(gauge_info)

        return "\n".join(result)

    def _format_endings(self, endings: List[FinalEnding]) -> str:
        """엔딩 정보를 프롬프트용 문자열로 포맷팅"""
        if not endings:
            return "등록된 엔딩 없음"

        result = []
        for e in endings:
            ending_info = f"""• [{e.get('type', 'unknown')}] {e.get('title', '제목없음')}
  - 조건: {e.get('condition', '불명')}"""
            result.append(ending_info)

        return "\n".join(result)

    def _calculate_current_gauges(self, state: Dict, choice_taken: Optional[StoryChoice]) -> Dict[str, int]:
        """현재 게이지 상태 계산"""
        # 초기값 설정 (모든 게이지 50에서 시작)
        current = state.get("current_gauges", {})
        if not current:
            for g in state.get("context", {}).get("gauges", []):
                current[g.get("id", g.get("name", "unknown"))] = 50
        else:
            current = current.copy()

        # 선택에 따른 게이지 변화 적용
        if choice_taken and "gauge_changes" in choice_taken:
            for gauge_id, change in choice_taken["gauge_changes"].items():
                if gauge_id in current:
                    current[gauge_id] = max(0, min(100, current[gauge_id] + change))
                else:
                    current[gauge_id] = max(0, min(100, 50 + change))

        return current

    # --- LangGraph 내부 로직 (Manager) ---
    def _plan_next_step(self, state):
        """
        Map-Reduce 패턴을 사용한 트리 분기 로직

        - 각 노드의 선택지마다 새로운 자식 노드를 생성
        - 최대 깊이에 도달하면 종료
        - 선택지가 없는 노드(엔딩)는 더 이상 분기하지 않음
        """
        nodes = state.get("nodes", [])
        max_depth = state.get("max_depth", 5)
        context = state.get("context", {})
        current_gauges = state.get("current_gauges", {})

        # 아직 노드가 없으면 루트 노드 생성을 위해 초기 상태 반환
        if not nodes:
            return [Send("generate_node", {
                "context": context,
                "max_depth": max_depth,
                "current_gauges": current_gauges
            })]

        # 가장 최근에 생성된 노드들 찾기 (같은 깊이의 노드들)
        # LangGraph의 Map-Reduce에서는 병렬로 생성된 노드들이 한 번에 추가됨
        if len(nodes) == 1:
            latest_nodes = nodes
        else:
            # 가장 깊은 depth의 노드들을 찾음
            max_current_depth = max(n["depth"] for n in nodes)
            latest_nodes = [n for n in nodes if n["depth"] == max_current_depth]

        # 최대 깊이 체크
        if latest_nodes and latest_nodes[0]["depth"] > max_depth:
            print(f"🏁 최대 깊이 {max_depth} 도달. 트리 생성 완료.")
            return END

        # 각 최신 노드의 선택지에 대해 자식 노드 생성 태스크 생성
        tasks = []
        for node in latest_nodes:
            choices = node.get("choices", [])

            if not choices:
                # 선택지가 없는 노드 (엔딩 또는 에러)는 스킵
                continue

            for choice_idx, choice in enumerate(choices):
                # 이 선택지를 선택했을 때의 자식 노드 생성 태스크
                task = {
                    "task": {
                        "depth": node["depth"] + 1,
                        "parent_node": node,
                        "choice_taken": choice,
                        "choice_index": choice_idx
                    },
                    "context": context,
                    "max_depth": max_depth,
                    "current_gauges": current_gauges
                }
                tasks.append(Send("generate_node", task))
                print(f"  📝 태스크 예약: depth={node['depth']+1}, parent={node['id']}, choice={choice_idx}")

        if not tasks:
            print("🏁 더 이상 생성할 노드가 없습니다. 트리 생성 완료.")
            return END

        print(f"🔀 {len(tasks)}개의 분기 노드 생성 시작...")
        return tasks

    # 유틸리티
    def _parse_json(self, content: str) -> Dict:
        """LLM 응답에서 JSON을 안전하게 파싱"""
        try:
            # 먼저 LangChain 파서 시도
            return self.json_parser.parse(content)
        except Exception:
            pass

        # 직접 JSON 추출 시도
        try:
            # ```json ... ``` 블록 추출
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)

            # { } 블록 직접 추출 (가장 큰 JSON 객체 찾기)
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0).strip()
                return json.loads(json_str)

            # 직접 파싱 시도
            return json.loads(content.strip())

        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON 파싱 실패: {e}")
            # 디버깅을 위해 응답의 일부 출력
            preview = content[:300] if len(content) > 300 else content
            print(f"  📄 응답 미리보기: {preview}")

        return {}

    async def _generate_summary(self, novel_text: str) -> str:
        """소설 텍스트 요약 생성 - 청크 분할 후 통합 요약"""
        chunk_size = 20000  # 각 청크 크기

        if len(novel_text) <= chunk_size:
            # 짧은 텍스트는 바로 요약
            prompt = f"""다음 소설 텍스트를 500자 내외로 요약해주세요.
핵심 줄거리, 주제, 갈등 구조, 결말을 포함해야 합니다.

[소설 텍스트]
{novel_text}
"""
            response = await self.llm.ainvoke(prompt)
            return response.content

        # 긴 텍스트는 청크로 나눠서 각각 요약
        print(f"  📚 긴 텍스트 감지 ({len(novel_text):,}자), 청크 분할 요약 시작...")

        chunks = []
        for i in range(0, len(novel_text), chunk_size):
            chunks.append(novel_text[i:i + chunk_size])

        print(f"  📦 {len(chunks)}개 청크로 분할")

        # 각 청크 요약
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"    [{i+1}/{len(chunks)}] 청크 요약 중...")
            prompt = f"""다음은 소설의 {i+1}번째 부분입니다. 이 부분의 핵심 내용을 200자 내외로 요약해주세요.
주요 사건, 등장인물의 행동, 갈등을 포함하세요.

[텍스트]
{chunk}
"""
            response = await self.llm.ainvoke(prompt)
            chunk_summaries.append(f"[파트 {i+1}] {response.content}")

        # 청크 요약들을 통합하여 최종 요약
        print("  🔄 청크 요약 통합 중...")
        combined_summaries = "\n\n".join(chunk_summaries)

        final_prompt = f"""다음은 소설의 각 부분별 요약입니다. 이를 바탕으로 전체 소설을 500자 내외로 통합 요약해주세요.
핵심 줄거리, 주제, 갈등 구조, 결말을 포함해야 합니다.

[부분별 요약]
{combined_summaries}
"""
        response = await self.llm.ainvoke(final_prompt)
        print("  ✅ 통합 요약 완료")

        return response.content


# ==============================================================================
# 3. LangGraph 상태 정의 (State Definition)
# ==============================================================================

def merge_gauges(current: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    """게이지 상태 병합 (최신 값으로 업데이트)"""
    if not new:
        return current
    result = current.copy() if current else {}
    result.update(new)
    return result

class StoryGenerationState(TypedDict):
    nodes: Annotated[List[StoryNode], operator.add]
    context: Dict[str, Any]  # 캐릭터, 소설요약, 게이지, 엔딩, 가이드 등 모든 정보
    max_depth: int
    current_gauges: Annotated[Dict[str, int], merge_gauges]  # 현재 게이지 상태
