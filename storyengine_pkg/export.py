from typing import Dict
import json


def export_to_markdown(result: Dict, output_path: str = "story_export.md") -> str:
    """스토리를 마크다운 형식으로 내보내기"""
    lines = []

    # 메타데이터
    meta = result.get("metadata", {})
    lines.append(f"# 인터랙티브 스토리")
    lines.append(f"\n- 에피소드: {meta.get('total_episodes', 0)}개")
    lines.append(f"- 노드: {meta.get('total_nodes', 0)}개")
    lines.append(f"- 게이지: {', '.join(meta.get('gauges', []))}")
    lines.append("")

    # 최종 엔딩
    context = result.get("context", {})
    lines.append("## 최종 엔딩")
    for ending in context.get("final_endings", []):
        lines.append(f"\n### [{ending.get('type', '?')}] {ending.get('title', '?')}")
        lines.append(f"- 조건: `{ending.get('condition', '?')}`")
        lines.append(f"- {ending.get('summary', '')}")
    lines.append("")

    # 에피소드별
    for episode in result.get("episodes", []):
        lines.append(f"\n## 에피소드 {episode.get('order', '?')}: {episode.get('title', '?')}")
        lines.append(f"\n**테마**: {episode.get('theme', '?')}")
        lines.append(f"\n**설명**: {episode.get('description', '?')}")

        # 도입부
        if episode.get("intro_text"):
            lines.append(f"\n### 도입부")
            lines.append(f"\n{episode.get('intro_text', '')}")

        # 엔딩
        lines.append(f"\n### 에피소드 엔딩")
        for ending in episode.get("endings", []):
            changes = ending.get("gauge_changes", {})
            change_str = ", ".join([f"{k}: {'+' if v > 0 else ' '}{v}" for k, v in changes.items()])
            lines.append(f"\n#### {ending.get('title', '?')}")
            lines.append(f"- 조건: `{ending.get('condition', '?')}`")
            lines.append(f"- 게이지: {change_str}")
            lines.append(f"- {ending.get('text', '')}")

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def export_to_html(result: Dict, output_path: str = "story_export.html") -> str:
    """스토리를 HTML 형식으로 내보내기"""
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head>")
    html.append("<meta charset='utf-8'>")
    html.append("<title>인터랙티브 스토리</title>")
    html.append("<style>")
    html.append("body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }")
    html.append(".episode { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }")
    html.append(".intro { background: #f5f5f5; padding: 15px; border-radius: 5px; }")
    html.append(".ending { background: #e8f4e8; padding: 10px; margin: 10px 0; border-radius: 5px; }")
    html.append(".gauge { color: #666; font-size: 0.9em; }")
    html.append("</style>")
    html.append("</head><body>")

    meta = result.get("metadata", {})
    html.append(f"<h1>인터랙티브 스토리</h1>")
    html.append(f"<p>에피소드: {meta.get('total_episodes', 0)}개 | 노드: {meta.get('total_nodes', 0)}개</p>")

    for episode in result.get("episodes", []):
        html.append(f"<div class='episode'>")
        html.append(f"<h2>에피소드 {episode.get('order', '?')}: {episode.get('title', '?')}</h2>")
        html.append(f"<p><strong>테마:</strong> {episode.get('theme', '?')}</p>")

        if episode.get("intro_text"):
            html.append(f"<div class='intro'><h3>도입부</h3>")
            html.append(f"<p>{episode.get('intro_text', '').replace(chr(10), '<br>')}</p></div>")

        html.append(f"<h3>엔딩</h3>")
        for ending in episode.get("endings", []):
            changes = ending.get("gauge_changes", {})
            change_str = ", ".join([f"{k}: {'+' if v > 0 else ' '}{v}" for k, v in changes.items()])
            html.append(f"<div class='ending'>")
            html.append(f"<strong>{ending.get('title', '?')}</strong>")
            html.append(f"<p class='gauge'>조건: {ending.get('condition', '?')} | {change_str}</p>")
            html.append(f"<p>{ending.get('text', '')}</p></div>")

        html.append("</div>")

    html.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    return output_path


def export_for_game_engine(result: Dict, output_path: str = "story_game.json") -> str:
    """게임 엔진용 간소화된 JSON 내보내기"""
    export_data = {
        "gauges": result.get("context", {}).get("gauges", []),
        "final_endings": result.get("context", {}).get("final_endings", []),
        "episodes": []
    }

    for episode in result.get("episodes", []):
        ep_data = {
            "id": episode.get("id"),
            "title": episode.get("title"),
            "order": episode.get("order"),
            "intro_text": episode.get("intro_text", ""),
            "endings": episode.get("endings", []),
            "nodes": []
        }

        # 노드 간소화
        for node in episode.get("nodes", []):
            node_data = {
                "id": node.get("id"),
                "parent_id": node.get("parent_id"),
                "text": node.get("text"),
                "choices": [{"text": c.get("text"), "tags": c.get("tags", [])} for c in node.get("choices", [])]
            }
            ep_data["nodes"].append(node_data)

        export_data["episodes"].append(ep_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return output_path