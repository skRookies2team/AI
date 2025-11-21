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
    max_depth: int = 3
) -> Dict:
    """
    에피소드 기반 인터랙티브 스토리 생성 파이프라인 (API용)

    Args:
        api_key: OpenAI API 키
        novel_text: 원본 소설 텍스트
        selected_gauge_ids: 선택된 게이지 ID 리스트 (2개)
        num_episodes: 에피소드 개수 (기본값: 4)
        max_depth: 에피소드별 트리 최대 깊이 (기본값: 3, 범위: 2~5)

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
    print("\n🏁 [4단계] 최종 엔딩 설계 중...")
    final_endings = await director.design_final_endings(
        novel_summary,
        selected_gauges,
        "다양한 결말을 포함해주세요 (해피엔딩, 비극, 열린 결말 등)"
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
        episode_endings = await director.design_episode_endings(ep_template, selected_gauges, num_endings=3)

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
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        novel_text = f.read()
                    print(f"  ✅ 파일 로드 완료: {len(novel_text):,}자")
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

            # 스토리 생성
            result = await main_flow(
                api_key=API_KEY,
                novel_text=novel_text,
                selected_gauge_ids=selected_gauge_ids,
                num_episodes=num_episodes,
                max_depth=max_depth
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
