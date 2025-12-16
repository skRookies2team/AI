import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

from storyengine_pkg import (
    InteractiveStoryDirector,
    save_episode_story,
    Episode,
)


async def main_flow(
    api_key: str,
    novel_text: str,
    selected_gauge_ids: List[str],
    num_episodes: int = 4,
    max_depth: int = 3,
    ending_config: Optional[Dict[str, int]] = None,
    num_episode_endings: int = 3
) -> Dict:
    """
    에피소드 기반 인터랙티브 스토리 생성 파이프라인 (API용)

    Args:
        api_key: OpenAI API 키
        novel_text: 원본 소설 텍스트
        selected_gauge_ids: 선택된 게이지 ID 리스트 (2개)
        num_episodes: 에피소드 개수 (기본값: 4)
        max_depth: 에피소드별 트리 최대 깊이 (기본값: 3, 범위: 2~5)
        ending_config: 최종 엔딩 타입별 개수 설정
            예: {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}
            지원 타입: happy, tragic, neutral, open, bad, bittersweet
        num_episode_endings: 에피소드별 엔딩 개수 (기본값: 3)

    Returns:
        생성된 에피소드 리스트 (각 에피소드에 노드와 엔딩 포함)
    """
    print("=" * 60)
    print("🎬 에피소드 기반 인터랙티브 스토리 생성 파이프라인")
    print("=" * 60)

    director = InteractiveStoryDirector(api_key=api_key)

    # ========================================
    # 1단계: 소설 요약 생성
    # ========================================
    print("\n📝 [1단계] 소설 요약 생성 중...")
    novel_summary = await director._generate_summary(novel_text)
    print(f"  ✅ 요약 완료 ({len(novel_summary)}자)")

    # ========================================
    # 2단계: 등장인물 추출
    # ========================================
    print("\n👥 [2단계] 등장인물 분석 중...")
    characters = await director.extract_characters(novel_text)
    print(f"  ✅ {len(characters)}명의 캐릭터 추출 완료")
    for char in characters:
        print(f"    • {char.get('name', '이름없음')}")

    # ========================================
    # 3단계: 게이지 시스템 설계
    # ========================================
    print("\n📊 [3단계] 게이지 시스템 설계 중...")
    gauges = await director.suggest_gauges(novel_summary)
    print(f"  ✅ {len(gauges)}개의 게이지 제안됨")

    # 선택된 게이지 필터링
    selected_gauges = [g for g in gauges if g.get('id') in selected_gauge_ids]

    # 선택된 게이지가 부족하면 앞에서부터 채움
    if len(selected_gauges) < 2:
        for g in gauges:
            if g not in selected_gauges:
                selected_gauges.append(g)
            if len(selected_gauges) >= 2:
                break

    print(f"  📌 선택된 게이지: {[g.get('name') for g in selected_gauges]}")
    print(f"  🌳 트리 깊이: {max_depth}")

    # ========================================
    # 4단계: 최종 엔딩 설계 (게이지 누적 기반)
    # ========================================
    if ending_config is None:
        ending_config = {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}
    total_endings = sum(ending_config.values())
    print(f"\n🏁 [4단계] 최종 엔딩 설계 중 ({total_endings}개)...")
    final_endings = await director.design_final_endings(
        novel_summary,
        selected_gauges,
        ending_config=ending_config
    )
    print(f"  ✅ {len(final_endings)}개의 최종 엔딩 설계 완료")
    for e in final_endings:
        print(f"    • [{e.get('type', '?')}] {e.get('title', '제목없음')}")
        print(f"      조건: {e.get('condition', '?')}")

    # ========================================
    # 5단계: 에피소드 분할
    # ========================================
    print(f"\n📚 [5단계] 에피소드 분할 중 ({num_episodes}개)...")
    episode_templates = await director.split_into_episodes(novel_summary, characters, num_episodes)

    # ========================================
    # 6단계: 각 에피소드별 트리 및 엔딩 생성
    # ========================================
    print("\n🌳 [6단계] 에피소드별 스토리 생성 시작...")

    completed_episodes: List[Episode] = []

    for ep_template in episode_templates:
        ep_id = ep_template.get('id', f"ep{ep_template.get('order', 0)}")
        ep_title = ep_template.get('title', '제목없음')

        print(f"\n  📖 에피소드 {ep_template.get('order', '?')}: {ep_title}")

        # 🌟 에피소드 도입부 생성
        intro_text = await director.generate_episode_intro(ep_template, characters, novel_summary)

        # 컨텍스트 구성 (에피소드 정보 포함)
        context = {
            "characters": characters,
            "gauges": selected_gauges,
            "endings": final_endings,
            "novel_summary": novel_summary,
            "episode_id": ep_id,
            "episode_info": ep_template,
            "intro_text": intro_text  # 도입부 컨텍스트 전달
        }

        # 에피소드 트리 생성
        episode_nodes = await director.generate_full_tree(context, max_depth=max_depth)

        # 에피소드 엔딩 설계
        episode_endings = await director.design_episode_endings(ep_template, selected_gauges, num_endings=num_episode_endings)

        # 완성된 에피소드 조립
        completed_episode: Episode = {
            "id": ep_id,
            "title": ep_title,
            "order": ep_template.get('order', 0),
            "description": ep_template.get('description', ''),
            "theme": ep_template.get('theme', ''),
            "intro_text": intro_text,  # 🌟 도입부 포함
            "nodes": episode_nodes,
            "endings": episode_endings
        }

        completed_episodes.append(completed_episode)
        print(f"    ✅ 에피소드 완료: 도입부 + {len(episode_nodes)}개 노드, {len(episode_endings)}개 엔딩")

    # ========================================
    # 7단계: 결과 저장
    # ========================================
    print("\n💾 [7단계] 결과 저장 중...")

    # 전체 결과 구성
    result = {
        "metadata": {
            "total_episodes": len(completed_episodes),
            "total_nodes": sum(len(ep.get("nodes", [])) for ep in completed_episodes),
            "gauges": [g.get("name") for g in selected_gauges],
            "character_count": len(characters)
        },
        "context": {
            "novel_summary": novel_summary,
            "characters": characters,
            "gauges": selected_gauges,
            "final_endings": final_endings
        },
        "episodes": completed_episodes
    }

    output_path = save_episode_story(result)
    print(f"  ✅ 저장 완료: {output_path}")

    print("\n" + "=" * 60)
    print("🎉 에피소드 기반 스토리 생성 파이프라인 완료!")
    print(f"📊 총 {len(completed_episodes)}개 에피소드, {result['metadata']['total_nodes']}개 노드 생성")
    print("=" * 60)

    return result


async def get_gauges(api_key: str, novel_text: str) -> Dict:
    """
    게이지 제안만 받아오는 함수 (프론트엔드에서 게이지 선택 UI용)

    Args:
        api_key: OpenAI API 키
        novel_text: 원본 소설 텍스트

    Returns:
        {
            "summary": 소설 요약,
            "characters": 캐릭터 리스트,
            "gauges": 제안된 게이지 리스트
        }
    """
    director = InteractiveStoryDirector(api_key=api_key)

    # 요약 생성
    novel_summary = await director._generate_summary(novel_text)

    # 캐릭터 추출
    characters = await director.extract_characters(novel_text)

    # 게이지 제안
    gauges = await director.suggest_gauges(novel_summary)

    return {
        "summary": novel_summary,
        "characters": characters,
        "gauges": gauges
    }


async def finalize_analysis(
    api_key: str,
    novel_summary: str,
    selected_gauges: List[Dict],
    ending_config: Optional[Dict] = None
) -> Dict:
    """
    사용자가 선택한 게이지를 기반으로 최종 엔딩을 생성하는 함수

    Args:
        api_key: OpenAI API 키
        novel_summary: 소설 요약
        selected_gauges: 사용자가 선택한 게이지 리스트 (2-3개)
        ending_config: 엔딩 타입별 개수 설정 (기본값: {"happy": 2, "tragic": 1, "neutral": 1, "open": 1})

    Returns:
        {
            "finalEndings": 최종 엔딩 리스트
        }
    """
    director = InteractiveStoryDirector(api_key=api_key)

    if ending_config is None:
        ending_config = {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}

    # 선택된 게이지만 사용하여 최종 엔딩 설계
    final_endings = await director.design_final_endings(
        novel_summary,
        selected_gauges,
        ending_config=ending_config
    )

    return {
        "finalEndings": final_endings
    }


async def regenerate_subtree(
    api_key: str,
    parent_node: Dict,
    novel_context: str,
    selected_gauge_ids: List[str],
    current_depth: int,
    max_depth: int,
    episode_title: str = "",
    previous_choices: List[str] = None,
    cached_summary: str = None,
    cached_characters_json: str = None,
    cached_gauges_json: str = None
) -> Dict:
    """
    수정된 부모 노드를 기반으로 하위 서브트리를 재생성합니다.

    Args:
        api_key: OpenAI API 키
        parent_node: 수정된 부모 노드 정보 (nodeId, text, choices, situation, npcEmotions, tags, depth)
        novel_context: 원작 소설 텍스트
        selected_gauge_ids: 선택된 게이지 ID 리스트
        current_depth: 부모 노드의 현재 깊이
        max_depth: 트리의 최대 깊이
        episode_title: 에피소드 제목
        previous_choices: 이전 선택 경로

    Returns:
        {
            "status": "success",
            "message": "Subtree regenerated",
            "regeneratedNodes": [...],
            "totalNodesRegenerated": 개수
        }
    """
    print("=" * 60)
    print("🔄 서브트리 재생성 시작")
    print("=" * 60)
    print(f"  부모 노드: {parent_node.get('nodeId')}")
    print(f"  현재 깊이: {current_depth}/{max_depth}")
    print(f"  부모 선택지 개수: {len(parent_node.get('choices', []))}")

    if previous_choices is None:
        previous_choices = []

    director = InteractiveStoryDirector(api_key=api_key)

    # 1. 소설 요약 및 캐릭터 정보 준비 (캐시 활용)
    if cached_summary and cached_characters_json:
        print("\n📝 [1단계] 캐시된 분석 결과 사용 (성능 최적화)")
        novel_summary = cached_summary
        import json
        characters = json.loads(cached_characters_json)
        print(f"  ✅ 캐시 활용: 요약 & {len(characters)}명의 캐릭터")
    else:
        print("\n📝 [1단계] 소설 분석 중...")
        novel_summary = await director._generate_summary(novel_context)
        characters = await director.extract_characters(novel_context)
        print(f"  ✅ 요약 완료, {len(characters)}명의 캐릭터 추출")

    # 2. 게이지 정보 준비 (캐시 활용)
    if cached_gauges_json:
        print("\n📊 [2단계] 캐시된 게이지 정보 사용")
        import json
        all_gauges = json.loads(cached_gauges_json)
        print(f"  ✅ 캐시 활용: {len(all_gauges)}개 게이지")
    else:
        print("\n📊 [2단계] 게이지 시스템 로드 중...")
        all_gauges = await director.suggest_gauges(novel_summary)
        print(f"  ✅ {len(all_gauges)}개 게이지 생성")

    selected_gauges = [g for g in all_gauges if g.get('id') in selected_gauge_ids]

    if len(selected_gauges) < len(selected_gauge_ids):
        # ID가 일치하지 않는 경우 경고 및 에러 처리
        found_ids = {g.get('id') for g in selected_gauges}
        missing_ids = set(selected_gauge_ids) - found_ids
        print(f"  ⚠️ Warning: Requested gauge IDs not found: {missing_ids}")
        print(f"  ⚠️ Available gauge IDs: {[g.get('id') for g in all_gauges]}")

        # 누락된 ID에 대해 사용 가능한 게이지로 대체 (fallback)
        for g in all_gauges:
            if g not in selected_gauges and len(selected_gauges) < len(selected_gauge_ids):
                selected_gauges.append(g)
                print(f"  🔄 Fallback: Using gauge '{g.get('name')}' (id: {g.get('id')})")

    print(f"  📌 선택된 게이지: {[g.get('name') for g in selected_gauges]}")

    # 3. 컨텍스트 구성
    context = {
        "characters": characters,
        "gauges": selected_gauges,
        "endings": [],  # 서브트리 재생성에서는 엔딩 불필요
        "novel_summary": novel_summary,
        "episode_title": episode_title
    }

    # 4. 부모 노드의 각 선택지에 대해 자식 노드 생성
    print(f"\n🌳 [3단계] 자식 노드 생성 중...")
    regenerated_nodes = []

    parent_choices = parent_node.get('choices', [])

    for choice_idx, choice_text in enumerate(parent_choices):
        print(f"\n  선택지 {choice_idx + 1}/{len(parent_choices)}: '{choice_text}'")

        # 자식 노드 트리 생성 (depth는 current_depth + 1부터 시작)
        child_nodes = await _generate_child_subtree(
            director=director,
            parent_text=parent_node.get('text'),
            choice_text=choice_text,
            current_depth=current_depth + 1,
            max_depth=max_depth,
            context=context
        )

        if child_nodes and len(child_nodes) > 0:
            regenerated_nodes.append(child_nodes[0])  # 각 선택지의 루트 자식 노드
            print(f"    ✅ {_count_nodes(child_nodes[0])}개 노드 생성")
        else:
            print(f"    ⚠️ Warning: Failed to generate child nodes for choice '{choice_text}'")

    # 5. 결과 반환
    total_regenerated = sum(_count_nodes(node) for node in regenerated_nodes)

    print("\n" + "=" * 60)
    print(f"🎉 서브트리 재생성 완료!")
    print(f"📊 총 {total_regenerated}개 노드 생성")
    print("=" * 60)

    return {
        "status": "success",
        "message": "Subtree regenerated",
        "regeneratedNodes": regenerated_nodes,
        "totalNodesRegenerated": total_regenerated
    }


async def _generate_single_node(
    director: 'InteractiveStoryDirector',
    parent_text: str,
    choice_text: str,
    depth: int,
    max_depth: int,
    node_type: str,
    context: Dict
) -> Dict:
    """
    단일 노드를 LLM으로 생성합니다.
    """
    import uuid
    import json
    from langchain_core.messages import SystemMessage, HumanMessage

    # 캐릭터, 게이지, 엔딩 정보 포맷팅 (director의 메서드 활용)
    characters_info = director._format_characters(context.get("characters", []))
    gauges_info = director._format_gauges(context.get("gauges", []))

    system_prompt = f"""당신은 인터랙티브 소설 작가입니다. 주어진 컨텍스트를 바탕으로 스토리 노드를 생성합니다.

[소설 배경]
{context.get('novel_summary', '정보 없음')}

[등장인물]
{characters_info}

[게이지 시스템]
{gauges_info}

[현재 노드 정보]
- 깊이: {depth}/{max_depth}
- 노드 타입: {node_type}

[이전 스토리]
{parent_text}

[플레이어의 선택]
{choice_text}"""

    user_prompt = f"""위 컨텍스트를 바탕으로 다음 스토리 노드를 생성하세요.

[작성 요구사항]
1. **스토리 본문** (500-800자): 선택 이후의 상황을 생생하게 묘사. 캐릭터들의 대화와 행동 포함.
2. **디테일 정보**:
   - npc_emotions: 현재 등장하는 NPC들의 감정 상태
   - situation: 현재 상황 한 줄 요약
   - tags: 이 장면의 분위기/주제 태그 (1~3개)
3. **선택지** (2~4개, 상황에 맞게 판단):
   - 선택지 개수는 현재 상황의 복잡도에 따라 2~4개 중 적절히 결정
   - 선택지 텍스트는 플레이어 관점에서 1인칭으로 작성

{"⚠️ 이것은 엔딩 노드입니다. 스토리를 마무리하고 선택지는 빈 배열로 두세요." if node_type == "ending" else ""}

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "text": "스토리 본문...",
    "details": {{
        "npcEmotions": {{"캐릭터명": "감정"}},
        "situation": "상황 요약"
    }},
    "choices": [
        {{
            "text": "선택지 1",
            "tags": ["태그1", "태그2"],
            "immediate_reaction": "선택 1에 대한 즉각적인 반응..."
        }},
        {{
            "text": "선택지 2",
            "tags": ["태그3", "태그4"],
            "immediate_reaction": "선택 2에 대한 즉각적인 반응..."
        }}
    ]
}}"""

    try:
        response = await director.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        parsed = director._parse_json(response.content)

        # Ensure details is a dictionary
        details = parsed.get("details", {})
        if not isinstance(details, dict):
            details = {"situation": "Parsing error", "npcEmotions": {}}

        return {
            "text": parsed.get("text", "스토리 생성 실패"),
            "details": {
                "npcEmotions": details.get("npcEmotions", {}),
                "situation": details.get("situation", "")
            },
            "choices": parsed.get("choices", [])  # Should be a list of objects
        }
    except Exception as e:
        print(f"    ❌ 노드 생성 실패: {e}")
        return {
            "text": f"[오류로 인해 스토리를 생성할 수 없습니다: {str(e)}]",
            "details": {
                "npcEmotions": {},
                "situation": "오류 발생"
            },
            "choices": []
        }


async def _generate_child_subtree(
    director: 'InteractiveStoryDirector',
    parent_text: str,
    choice_text: str,
    current_depth: int,
    max_depth: int,
    context: Dict
) -> List[Dict]:
    """
    단일 선택지에 대한 서브트리를 재귀적으로 생성합니다.
    """
    import uuid

    # 노드 타입 결정
    if current_depth == max_depth:
        node_type = "ending"
    elif current_depth == max_depth - 1:
        node_type = "climax"
    else:
        node_type = "development"

    # LLM으로 자식 노드 생성
    node_data = await _generate_single_node(
        director=director,
        parent_text=parent_text,
        choice_text=choice_text,
        depth=current_depth,
        max_depth=max_depth,
        node_type=node_type,
        context=context
    )

    # 노드 구성
    node_id = f"node_{uuid.uuid4().hex[:8]}"
    child_node = {
        "id": node_id,
        "text": node_data.get("text", ""),
        "choices": node_data.get("choices", []),  # This is now the list of choice objects
        "depth": current_depth,
        "details": node_data.get("details", {}),  # Use the nested details object directly
        "children": []
    }

    # 재귀적으로 자식 노드의 자식들 생성 (max_depth 도달 전까지)
    if current_depth < max_depth and node_data.get("choices"):
        for sub_choice_obj in node_data.get("choices", []):
            # Pass the text of the choice object to the recursive call
            sub_choice_text = sub_choice_obj.get("text") if isinstance(sub_choice_obj, dict) else sub_choice_obj

            sub_children = await _generate_child_subtree(
                director=director,
                parent_text=child_node["text"],
                choice_text=sub_choice_text,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                context=context
            )
            if sub_children and len(sub_children) > 0:
                child_node["children"].append(sub_children[0])

    return [child_node]


def _count_nodes(node: Dict) -> int:
    """트리 노드 개수를 재귀적으로 계산"""
    if node is None:
        return 0
    count = 1
    children = node.get("children", [])
    if children:
        for child in children:
            count += _count_nodes(child)
    return count


# ============================================
# CLI 실행용 (터미널에서 직접 실행 시)
# ============================================
if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.environ.get("OPENAI_API_KEY")

    if not API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요.")
        exit(1)

    async def run():
        try:
            # 소설 파일 경로 입력
            print("\n📖 소설 텍스트 파일을 입력하세요.")
            while True:
                file_path = input("  → 파일 경로 (.txt): ").strip()
                if not file_path:
                    print("    ⚠️ 파일 경로를 입력하세요.")
                    continue

                # 여러 인코딩 시도
                encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-16']
                novel_text = None

                try:
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                novel_text = f.read()
                            print(f"  ✅ 파일 로드 완료 ({encoding}): {len(novel_text):,}자")
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue

                    if novel_text is None:
                        print(f"    ❌ 지원되는 인코딩으로 파일을 읽을 수 없습니다.")
                        continue

                    break
                except FileNotFoundError:
                    print(f"    ❌ 파일을 찾을 수 없습니다: {file_path}")
                except Exception as e:
                    print(f"    ❌ 파일 읽기 오류: {e}")

            # 1단계: 게이지 제안 받기
            print("\n📊 게이지 분석 중...")
            gauge_data = await get_gauges(API_KEY, novel_text)
            gauges = gauge_data["gauges"]

            # 게이지 선택
            print("\n🎯 사용할 게이지 2개를 선택하세요:")
            for i, g in enumerate(gauges):
                print(f"  [{i+1}] {g.get('name', '이름없음')} ({g.get('id')}): {g.get('meaning', '')}")

            selected_gauge_ids = []
            while len(selected_gauge_ids) < 2:
                try:
                    remaining = 2 - len(selected_gauge_ids)
                    choice = input(f"  → 게이지 번호 입력 ({remaining}개 더 선택): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(gauges):
                        gauge_id = gauges[idx].get('id')
                        if gauge_id not in selected_gauge_ids:
                            selected_gauge_ids.append(gauge_id)
                            print(f"    ✓ '{gauges[idx].get('name')}' 선택됨")
                        else:
                            print("    ⚠️ 이미 선택된 게이지입니다.")
                    else:
                        print(f"    ⚠️ 1~{len(gauges)} 사이의 번호를 입력하세요.")
                except ValueError:
                    print("    ⚠️ 숫자를 입력하세요.")

            # 트리 깊이 입력
            print("\n🌳 스토리 트리 깊이 설정")
            print("  - 깊이 2: 간단한 스토리 (약 7개 노드)")
            print("  - 깊이 3: 보통 스토리 (약 15~40개 노드)")
            print("  - 깊이 4: 복잡한 스토리 (약 40~120개 노드)")
            max_depth = 3
            while True:
                try:
                    depth_input = input("  → 트리 깊이 입력 (2~5, 기본값 3): ").strip()
                    if depth_input == "":
                        break
                    max_depth = int(depth_input)
                    if 2 <= max_depth <= 5:
                        break
                    else:
                        print("    ⚠️ 2~5 사이의 숫자를 입력하세요.")
                except ValueError:
                    print("    ⚠️ 숫자를 입력하세요.")

            # 에피소드 개수 입력
            num_episodes = 3
            try:
                ep_input = input("  → 에피소드 개수 (기본값 3): ").strip()
                if ep_input:
                    num_episodes = int(ep_input)
            except ValueError:
                print("    ⚠️ 기본값 3 사용")

            # 엔딩 타입별 개수 설정
            print("\n🏁 최종 엔딩 타입별 개수 설정")
            print("  지원 타입: happy(행복), tragic(비극), neutral(중립), open(열린결말), bad(나쁜), bittersweet(씁쓸)")
            print("  (엔터를 누르면 기본값 사용: happy=2, tragic=1, neutral=1, open=1)")

            ending_config = {}
            ending_types = [
                ("happy", "행복한 엔딩"),
                ("tragic", "비극적인 엔딩"),
                ("neutral", "중립적인 엔딩"),
                ("open", "열린 결말"),
                ("bad", "나쁜 엔딩"),
                ("bittersweet", "씁쓸한 엔딩")
            ]

            use_default = input("  → 기본값 사용? (y/n, 기본값 y): ").strip().lower()
            if use_default != 'n':
                ending_config = {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}
                print("    ✓ 기본값 사용: happy=2, tragic=1, neutral=1, open=1")
            else:
                for etype, ename in ending_types:
                    try:
                        count = input(f"    → {ename} ({etype}) 개수 (기본값 0): ").strip()
                        if count:
                            ending_config[etype] = int(count)
                    except ValueError:
                        pass

                if not ending_config or sum(ending_config.values()) == 0:
                    ending_config = {"happy": 2, "tragic": 1, "neutral": 1, "open": 1}
                    print("    ⚠️ 유효한 입력 없음, 기본값 사용")

            print(f"    📌 엔딩 설정: {ending_config}")

            num_episode_endings = 3
            try:
                ep_ending_input = input("  → 에피소드별 엔딩 개수 (기본값 3): ").strip()
                if ep_ending_input:
                    num_episode_endings = int(ep_ending_input)
            except ValueError:
                print("    ⚠️ 기본값 3 사용")

            # 스토리 생성
            result = await main_flow(
                api_key=API_KEY,
                novel_text=novel_text,
                selected_gauge_ids=selected_gauge_ids,
                num_episodes=num_episodes,
                max_depth=max_depth,
                ending_config=ending_config,
                num_episode_endings=num_episode_endings
            )

            # 결과 요약 출력
            print(f"\n🎯 생성 완료!")
            print(f"   - 에피소드: {result['metadata']['total_episodes']}개")
            print(f"   - 노드: {result['metadata']['total_nodes']}개")
            print(f"   - 게이지: {', '.join(result['metadata']['gauges'])}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            raise

    asyncio.run(run())
