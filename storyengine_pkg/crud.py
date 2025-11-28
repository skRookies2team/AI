from typing import List, Dict

from .models import Episode, StoryChoice


def edit_node(episodes: List[Episode], episode_id: str, node_id: str, updates: Dict) -> bool:
    """
    노드 내용 수정

    Args:
        episodes: 에피소드 리스트
        episode_id: 대상 에피소드 ID
        node_id: 대상 노드 ID
        updates: 수정할 필드들 {"text": "...", "details": {...}, ...}

    Returns:
        성공 여부
    """
    for episode in episodes:
        if episode.get("id") == episode_id:
            for node in episode.get("nodes", []):
                if node.get("id") == node_id:
                    for key, value in updates.items():
                        if key in node:
                            node[key] = value
                    return True
    return False


def delete_node(episodes: List[Episode], episode_id: str, node_id: str) -> bool:
    """
    노드 삭제 (자식 노드들도 함께 삭제)

    Returns:
        성공 여부
    """
    for episode in episodes:
        if episode.get("id") == episode_id:
            nodes = episode.get("nodes", [])

            # 삭제할 노드와 자식 노드들의 ID 수집
            to_delete = {node_id}
            changed = True
            while changed:
                changed = False
                for node in nodes:
                    if node.get("parent_id") in to_delete and node.get("id") not in to_delete:
                        to_delete.add(node.get("id"))
                        changed = True

            # 노드 삭제
            original_count = len(nodes)
            episode["nodes"] = [n for n in nodes if n.get("id") not in to_delete]

            return len(episode["nodes"]) < original_count

    return False


def add_choice(episodes: List[Episode], episode_id: str, node_id: str, choice: StoryChoice) -> bool:
    """노드에 선택지 추가"""
    for episode in episodes:
        if episode.get("id") == episode_id:
            for node in episode.get("nodes", []):
                if node.get("id") == node_id:
                    if "choices" not in node:
                        node["choices"] = []
                    node["choices"].append(choice)
                    return True
    return False


def remove_choice(episodes: List[Episode], episode_id: str, node_id: str, choice_index: int) -> bool:
    """노드에서 선택지 제거"""
    for episode in episodes:
        if episode.get("id") == episode_id:
            for node in episode.get("nodes", []):
                if node.get("id") == node_id:
                    choices = node.get("choices", [])
                    if 0 <= choice_index < len(choices):
                        choices.pop(choice_index)
                        return True
    return False


def update_episode_ending(episodes: List[Episode], episode_id: str, ending_id: str, updates: Dict) -> bool:
    """에피소드 엔딩 수정"""
    for episode in episodes:
        if episode.get("id") == episode_id:
            for ending in episode.get("endings", []):
                if ending.get("id") == ending_id:
                    for key, value in updates.items():
                        ending[key] = value
                    return True
    return False


def update_intro_text(episodes: List[Episode], episode_id: str, new_intro: str) -> bool:
    """에피소드 도입부 수정"""
    for episode in episodes:
        if episode.get("id") == episode_id:
            episode["intro_text"] = new_intro
            return True
    return False