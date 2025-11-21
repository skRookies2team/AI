import json
from typing import List, Dict, Optional

from .models import StoryChoice, EpisodeEnding, FinalEnding, Episode, StoryNode


def save_episode_story(result: Dict, filename: str = "episode_story.json") -> str:
    """에피소드 기반 스토리를 JSON 파일로 저장"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return filename


def load_episode_story(filename: str = "episode_story.json") -> Dict:
    """저장된 에피소드 스토리 로드"""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_tag_scores(choices_made: List[StoryChoice]) -> Dict[str, int]:
    """선택한 선택지들의 태그를 누적하여 점수 계산"""
    scores = {}
    for choice in choices_made:
        for tag in choice.get("tags", []):
            scores[tag] = scores.get(tag, 0) + 1
    return scores


def evaluate_condition(condition: str, tag_scores: Dict[str, int]) -> bool:
    """
    태그 점수 기반 조건식 평가

    지원 형식:
    - "cooperative >= 2"
    - "trusting > doubtful"
    - "cooperative >= 2 AND trusting >= 1"
    - "doubtful >= 2 OR aggressive >= 2"
    - "default" (항상 True)
    """
    if condition == "default":
        return True

    # AND/OR로 분리
    if " AND " in condition:
        parts = condition.split(" AND ")
        return all(evaluate_condition(part.strip(), tag_scores) for part in parts)

    if " OR " in condition:
        parts = condition.split(" OR ")
        return any(evaluate_condition(part.strip(), tag_scores) for part in parts)

    # 단일 조건 평가
    operators = [">=", "<=", ">", "<", "=="]
    for op in operators:
        if op in condition:
            parts = condition.split(op)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()

                # 왼쪽 값
                left_val = tag_scores.get(left, 0)

                # 오른쪽 값 (숫자 또는 태그명)
                if right.isdigit():
                    right_val = int(right)
                else:
                    right_val = tag_scores.get(right, 0)

                # 연산자 적용
                if op == ">=":
                    return left_val >= right_val
                elif op == "<=":
                    return left_val <= right_val
                elif op == ">":
                    return left_val > right_val
                elif op == "<":
                    return left_val < right_val
                elif op == "==":
                    return left_val == right_val

    return False


def determine_episode_ending(choices_made: List[StoryChoice], endings: List[EpisodeEnding]) -> EpisodeEnding:
    """
    플레이어의 선택에 따라 에피소드 엔딩 결정

    Args:
        choices_made: 플레이어가 선택한 선택지 리스트
        endings: 가능한 에피소드 엔딩 리스트

    Returns:
        조건을 만족하는 엔딩 (없으면 default 엔딩)
    """
    tag_scores = calculate_tag_scores(choices_made)

    # 각 엔딩의 조건 확인 (default 제외하고 먼저 체크)
    for ending in endings:
        condition = ending.get("condition", "default")
        if condition != "default" and evaluate_condition(condition, tag_scores):
            return ending

    # default 엔딩 반환
    for ending in endings:
        if ending.get("condition") == "default":
            return ending

    # 아무것도 없으면 첫 번째 엔딩
    return endings[0] if endings else None


def calculate_final_ending(episode_results: List[Dict], final_endings: List[FinalEnding], initial_gauges: Dict[str, int] = None) -> Dict:
    """
    모든 에피소드를 거친 후 최종 엔딩 결정

    Args:
        episode_results: 각 에피소드에서 도달한 엔딩과 게이지 변화
        final_endings: 가능한 최종 엔딩 리스트
        initial_gauges: 초기 게이지 값 (기본 50)

    Returns:
        최종 게이지 상태와 결정된 엔딩
    """
    # 초기 게이지 설정
    if initial_gauges is None:
        gauges = {}
    else:
        gauges = initial_gauges.copy()

    # 에피소드별 게이지 변화 누적
    for result in episode_results:
        ending = result.get("ending", {})
        changes = ending.get("gauge_changes", {})
        for gauge_id, change in changes.items():
            if gauge_id not in gauges:
                gauges[gauge_id] = 50
            gauges[gauge_id] = max(0, min(100, gauges[gauge_id] + change))

    # 최종 엔딩 결정
    for ending in final_endings:
        condition = ending.get("condition", "default")
        if condition != "default" and evaluate_gauge_condition(condition, gauges):
            return {"gauges": gauges, "ending": ending}

    # default 엔딩
    for ending in final_endings:
        if ending.get("condition") == "default":
            return {"gauges": gauges, "ending": ending}

    return {"gauges": gauges, "ending": final_endings[0] if final_endings else None}


def evaluate_gauge_condition(condition: str, gauges: Dict[str, int]) -> bool:
    """게이지 기반 조건식 평가 (최종 엔딩용)"""
    return evaluate_condition(condition, gauges)


def load_novel_from_file(file_path: str, encoding: str = "utf-8") -> str:
    """소설 파일 로드"""
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def save_story_tree(nodes: List[StoryNode], context: Dict, filename: str = "story_tree.json") -> str:
    """생성된 스토리 트리를 JSON 파일로 저장 (레거시)"""
    output = {
        "metadata": {
            "total_nodes": len(nodes),
            "max_depth": max(n.get("depth", 0) for n in nodes) if nodes else 0,
            "gauges": [g.get("name") for g in context.get("gauges", [])],
            "character_count": len(context.get("characters", []))
        },
        "context": {
            "novel_summary": context.get("novel_summary", ""),
            "characters": context.get("characters", []),
            "gauges": context.get("gauges", []),
            "endings": context.get("endings", [])
        },
        "nodes": nodes
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return filename


def load_story_tree(filename: str = "story_tree.json") -> Dict:
    """저장된 스토리 트리 로드"""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def get_node_by_id(nodes: List[StoryNode], node_id: str) -> Optional[StoryNode]:
    """ID로 노드 검색"""
    for node in nodes:
        if node.get("id") == node_id:
            return node
    return None


def get_children(nodes: List[StoryNode], parent_id: str) -> List[StoryNode]:
    """특정 노드의 자식 노드들 반환"""
    return [n for n in nodes if n.get("parent_id") == parent_id]


def get_path_to_node(nodes: List[StoryNode], node_id: str) -> List[StoryNode]:
    """루트에서 특정 노드까지의 경로 반환"""
    path = []
    current = get_node_by_id(nodes, node_id)

    while current:
        path.insert(0, current)
        parent_id = current.get("parent_id")
        if parent_id:
            current = get_node_by_id(nodes, parent_id)
        else:
            break

    return path


def print_story_path(nodes: List[StoryNode], target_node_id: str):
    """특정 노드까지의 스토리 경로 출력"""
    path = get_path_to_node(nodes, target_node_id)

    print("\n📖 스토리 경로:")
    print("-" * 40)

    for i, node in enumerate(path):
        print(f"\n[깊이 {node.get('depth', '?')}] {node.get('node_type', 'normal').upper()}")
        print(node.get("text", "")[:200] + "..." if len(node.get("text", "")) > 200 else node.get("text", ""))

        if node.get("choices") and i < len(path) - 1:
            # 다음 노드로 가기 위해 선택된 선택지 표시
            next_node = path[i + 1]
            for choice in node["choices"]:
                print(f"  → {choice.get('text', '?')}")