from typing import List, Dict

from .models import Episode, FinalEnding


def validate_gauge_balance(episodes: List[Episode], final_endings: List[FinalEnding], initial_value: int = 50) -> Dict:
    """
    모든 최종 엔딩에 도달 가능한지 검증

    Returns:
        {
            "is_balanced": bool,
            "reachable_endings": [...],
            "unreachable_endings": [...],
            "gauge_ranges": {"hope": {"min": 20, "max": 80}, ...},
            "recommendations": [...]
        }
    """
    # 모든 에피소드 엔딩 조합의 게이지 범위 계산
    gauge_ranges = {}

    # 각 게이지의 최소/최대 가능 값 계산
    for episode in episodes:
        for ending in episode.get("endings", []):
            changes = ending.get("gauge_changes", {})
            for gauge_id, change in changes.items():
                if gauge_id not in gauge_ranges:
                    gauge_ranges[gauge_id] = {"min": 0, "max": 0}
                if change > 0:
                    gauge_ranges[gauge_id]["max"] += change
                else:
                    gauge_ranges[gauge_id]["min"] += change

    # 초기값 적용
    for gauge_id in gauge_ranges:
        gauge_ranges[gauge_id]["min"] = max(0, initial_value + gauge_ranges[gauge_id]["min"])
        gauge_ranges[gauge_id]["max"] = min(100, initial_value + gauge_ranges[gauge_id]["max"])

    # 각 최종 엔딩 도달 가능 여부 확인
    reachable = []
    unreachable = []
    recommendations = []

    for ending in final_endings:
        condition = ending.get("condition", "default")
        if condition == "default":
            reachable.append(ending)
            continue

        # 조건 파싱하여 도달 가능 여부 확인
        is_reachable = check_condition_reachability(condition, gauge_ranges)

        if is_reachable:
            reachable.append(ending)
        else:
            unreachable.append(ending)
            recommendations.append(
                f"'{ending.get('title', '?')}' 엔딩 도달 불가: 조건 '{condition}'을 만족할 수 없음. "
                f"게이지 범위 조정 필요."
            )

    return {
        "is_balanced": len(unreachable) == 0,
        "reachable_endings": [e.get("title") for e in reachable],
        "unreachable_endings": [e.get("title") for e in unreachable],
        "gauge_ranges": gauge_ranges,
        "recommendations": recommendations
    }


def check_condition_reachability(condition: str, gauge_ranges: Dict) -> bool:
    """조건이 게이지 범위 내에서 만족 가능한지 확인"""
    if condition == "default":
        return True

    # AND 조건
    if " AND " in condition:
        parts = condition.split(" AND ")
        return all(check_condition_reachability(p.strip(), gauge_ranges) for p in parts)

    # OR 조건
    if " OR " in condition:
        parts = condition.split(" OR ")
        return any(check_condition_reachability(p.strip(), gauge_ranges) for p in parts)

    # 단일 조건 파싱
    operators = [">=", "<=", ">", "<", "=="]
    for op in operators:
        if op in condition:
            parts = condition.split(op)
            if len(parts) == 2:
                gauge_id = parts[0].strip()
                threshold = int(parts[1].strip())

                if gauge_id not in gauge_ranges:
                    return False

                range_min = gauge_ranges[gauge_id]["min"]
                range_max = gauge_ranges[gauge_id]["max"]

                if op == ">=":
                    return range_max >= threshold
                elif op == "<=":
                    return range_min <= threshold
                elif op == ">":
                    return range_max > threshold
                elif op == "<":
                    return range_min < threshold
                elif op == "==":
                    return range_min <= threshold <= range_max

    return False


def find_dead_ends(episodes: List[Episode]) -> List[Dict]:
    """선택지가 없는 비정상 노드(엔딩이 아닌) 탐지"""
    dead_ends = []

    for episode in episodes:
        for node in episode.get("nodes", []):
            # 엔딩 노드가 아닌데 선택지가 없는 경우
            if node.get("node_type") != "ending" and not node.get("choices"):
                dead_ends.append({
                    "episode_id": episode.get("id"),
                    "node_id": node.get("id"),
                    "depth": node.get("depth"),
                    "text_preview": node.get("text", "")[:100] + "..."
                })

    return dead_ends


def check_tag_coverage(episodes: List[Episode]) -> Dict:
    """에피소드 엔딩의 태그 조건 커버리지 확인"""
    all_tags = set()
    used_tags = set()

    # 사용된 모든 태그 수집
    for episode in episodes:
        for node in episode.get("nodes", []):
            for choice in node.get("choices", []):
                for tag in choice.get("tags", []):
                    all_tags.add(tag)

        # 엔딩 조건에서 사용된 태그
        for ending in episode.get("endings", []):
            condition = ending.get("condition", "")
            for tag in all_tags:
                if tag in condition:
                    used_tags.add(tag)

    unused_tags = all_tags - used_tags

    return {
        "all_tags": list(all_tags),
        "used_in_conditions": list(used_tags),
        "unused_tags": list(unused_tags),
        "coverage_rate": len(used_tags) / len(all_tags) if all_tags else 1.0
    }