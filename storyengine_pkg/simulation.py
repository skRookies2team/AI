from typing import List, Dict

from .models import Episode
from .utils import calculate_tag_scores, determine_episode_ending, calculate_final_ending


def simulate_playthrough(episode: Episode, choice_indices: List[int]) -> Dict:
    """
    특정 선택 경로로 에피소드 플레이 시뮬레이션

    Args:
        episode: 에피소드 데이터
        choice_indices: 각 노드에서의 선택지 인덱스 [0, 1, 0, ...]

    Returns:
        {
            "path": [노드들],
            "choices_made": [선택한 선택지들],
            "tag_scores": {"cooperative": 2, ...},
            "reached_ending": 엔딩 정보,
            "gauge_changes": {"hope": 10, ...}
        }
    """
    nodes = episode.get("nodes", [])
    endings = episode.get("endings", [])

    # 루트 노드 찾기
    root = None
    for node in nodes:
        if node.get("parent_id") is None:
            root = node
            break

    if not root:
        return {"error": "루트 노드를 찾을 수 없습니다"}

    path = [root]
    choices_made = []
    current_node = root

    # 선택 경로 따라가기
    for i, choice_idx in enumerate(choice_indices):
        choices = current_node.get("choices", [])
        if not choices:
            break

        if choice_idx >= len(choices):
            choice_idx = 0  # 범위 초과 시 첫 번째 선택

        selected_choice = choices[choice_idx]
        choices_made.append(selected_choice)

        # 다음 노드 찾기 (해당 선택지로 연결된 자식 노드)
        children = [n for n in nodes if n.get("parent_id") == current_node.get("id")]
        if choice_idx < len(children):
            current_node = children[choice_idx]
            path.append(current_node)
        else:
            break

    # 태그 점수 계산
    tag_scores = calculate_tag_scores(choices_made)

    # 엔딩 결정
    reached_ending = determine_episode_ending(choices_made, endings)

    return {
        "path": path,
        "choices_made": choices_made,
        "tag_scores": tag_scores,
        "reached_ending": reached_ending,
        "gauge_changes": reached_ending.get("gauge_changes", {}) if reached_ending else {}
    }


def simulate_full_game(result: Dict, episode_choices: List[List[int]]) -> Dict:
    """
    전체 게임 플레이 시뮬레이션

    Args:
        result: main_flow 결과
        episode_choices: 각 에피소드별 선택지 인덱스 [[0,1,0], [1,0], ...]

    Returns:
        {
            "episode_results": [...],
            "final_gauges": {"hope": 65, ...},
            "final_ending": 최종 엔딩 정보
        }
    """
    episodes = result.get("episodes", [])
    final_endings = result.get("context", {}).get("final_endings", [])

    episode_results = []

    for i, episode in enumerate(episodes):
        choices = episode_choices[i] if i < len(episode_choices) else []
        sim_result = simulate_playthrough(episode, choices)
        episode_results.append({
            "episode_id": episode.get("id"),
            "episode_title": episode.get("title"),
            "ending": sim_result.get("reached_ending"),
            "gauge_changes": sim_result.get("gauge_changes")
        })

    # 최종 엔딩 계산
    final_result = calculate_final_ending(episode_results, final_endings)

    return {
        "episode_results": episode_results,
        "final_gauges": final_result.get("gauges"),
        "final_ending": final_result.get("ending")
    }


def get_all_possible_endings(episode: Episode) -> List[Dict]:
    """에피소드에서 가능한 모든 엔딩 경로 분석"""
    nodes = episode.get("nodes", [])
    endings = episode.get("endings", [])

    # 모든 리프 노드까지의 경로 찾기
    def find_all_paths(node_id, current_path, all_paths):
        node = next((n for n in nodes if n.get("id") == node_id), None)
        if not node:
            return

        current_path = current_path + [node]
        children = [n for n in nodes if n.get("parent_id") == node_id]

        if not children or not node.get("choices"):
            # 리프 노드
            all_paths.append(current_path)
        else:
            for i, child in enumerate(children):
                find_all_paths(child.get("id"), current_path, all_paths)

    # 루트에서 시작
    root = next((n for n in nodes if n.get("parent_id") is None), None)
    if not root:
        return []

    all_paths = []
    find_all_paths(root.get("id"), [], all_paths)

    # 각 경로의 엔딩 분석
    results = []
    for path in all_paths:
        choices_made = []
        for i, node in enumerate(path[:-1]):
            choices = node.get("choices", [])
            if choices and i + 1 < len(path):
                # 다음 노드로 가는 선택지 찾기
                next_node = path[i + 1]
                children = [n for n in nodes if n.get("parent_id") == node.get("id")]
                for j, child in enumerate(children):
                    if child.get("id") == next_node.get("id") and j < len(choices):
                        choices_made.append(choices[j])
                        break

        tag_scores = calculate_tag_scores(choices_made)
        reached_ending = determine_episode_ending(choices_made, endings)

        results.append({
            "path_length": len(path),
            "tag_scores": tag_scores,
            "ending": reached_ending.get("title") if reached_ending else "없음",
            "gauge_changes": reached_ending.get("gauge_changes", {}) if reached_ending else {}
        })

    return results