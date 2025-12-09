import os
import json
from typing import List, Dict, Optional

from storyengine_pkg.director import InteractiveStoryDirector
from storyengine_pkg.models import (
    StoryConfig,
    InitialAnalysis,
    EpisodeModel,
)

async def generate_single_episode(
    api_key: str,
    initial_analysis: InitialAnalysis,
    story_config: StoryConfig,
    novel_context: str,
    current_episode_order: int,
    previous_episode_data: Optional[EpisodeModel]
) -> EpisodeModel:
    """
    Contains the core logic to generate one episode by calling the LLM.
    """
    director = InteractiveStoryDirector(api_key=api_key)

    # --- Determine the context for the LLM prompt ---
    if previous_episode_data is None:
        # This is the first episode.
        # 더 많은 컨텍스트 전달 (3000자 → 15000자)
        novel_excerpt = novel_context[:15000] if len(novel_context) > 15000 else novel_context
        excerpt_info = f"(showing first 15,000 characters of {len(novel_context):,} total)" if len(novel_context) > 15000 else "(complete text)"

        # 요약과 캐릭터 정보 추출
        summary_text = initial_analysis.summary if initial_analysis.summary else "ERROR: No summary provided"
        characters_text = json.dumps(initial_analysis.characters, ensure_ascii=False, indent=2) if initial_analysis.characters else "ERROR: No characters"

        context_prompt = f"""
        You are adapting an EXISTING novel into an interactive story game.

        ⚠️ CRITICAL: DO NOT CREATE A NEW STORY. You must use the EXACT plot below.
        ⚠️ FORBIDDEN: Using generic names like "캐릭터A", "캐릭터B", "주인공"
        ⚠️ REQUIRED: Use the SPECIFIC character names from the CHARACTER LIST below

        ═══════════════════════════════════════════════════════════════════
        📖 ORIGINAL STORY PLOT (YOU MUST FOLLOW THIS EXACTLY):
        ═══════════════════════════════════════════════════════════════════

        {summary_text}

        ═══════════════════════════════════════════════════════════════════
        👥 CHARACTER LIST (USE THESE EXACT NAMES IN YOUR EPISODE):
        ═══════════════════════════════════════════════════════════════════

        {characters_text}

        ═══════════════════════════════════════════════════════════════════
        📚 ORIGINAL NOVEL EXCERPT {excerpt_info}:
        ═══════════════════════════════════════════════════════════════════

        {novel_excerpt}

        ═══════════════════════════════════════════════════════════════════

        TASK: Create Episode {current_episode_order} of {story_config.num_episodes}
        THEMES: {', '.join(story_config.selected_gauge_ids)}

        🚫 ABSOLUTE PROHIBITIONS:
        1. DO NOT use generic character names (캐릭터A, 캐릭터B, 주인공, etc.)
        2. DO NOT create new plot events not in the summary above
        3. DO NOT change character names, relationships, or personalities
        4. DO NOT invent new characters not listed above
        5. DO NOT change the story's setting, time period, or core conflict

        ✅ MANDATORY REQUIREMENTS:
        1. Episode MUST start from an event described in the STORY PLOT above
        2. ALL character names MUST match the CHARACTER LIST exactly
        3. Scene descriptions MUST match the novel's setting and atmosphere
        4. Dialogue MUST reflect character personalities from the CHARACTER LIST
        5. Choices affect HOW events unfold, not WHAT events happen
        6. The plot timeline MUST follow the novel's sequence

        EXAMPLE (if story is Romeo and Juliet):
        ✅ CORRECT: "로미오는 캐플릿 가문의 무도회에 몰래 잠입했다..."
        ❌ WRONG: "캐릭터A는 어두운 밤에 갈등을 마주했다..."

        Before generating, ask yourself:
        - Did I use the EXACT character names from the list?
        - Is this scene from the actual novel plot?
        - Would someone who read the original novel recognize this?
        """
    else:
        # This is a subsequent episode (e.g., Ep 2, 3...).
        # A simple summary of the previous episode's ending.
        # A more advanced implementation could analyze the leaf nodes of the previous episode.
        previous_episode_summary = f"The previous episode, '{previous_episode_data.title}', ended."

        context_prompt = f"""
        The story continues. The previous episode ended like this: {previous_episode_summary}.
        Now, create episode {current_episode_order} of the story.
        Continue to incorporate the main themes: {', '.join(story_config.selected_gauge_ids)}.
        The overall story summary is: {initial_analysis.summary}
        The main characters are: {json.dumps(initial_analysis.characters, ensure_ascii=False, indent=2)}
        """

    # --- Construct the main LLM prompt ---
    llm_prompt = f"""
    You are an expert interactive story writer.
    {context_prompt}

    Generate the content for this episode, including a title, a starting node,
    a COMPLETE branching narrative tree up to EXACTLY depth {story_config.max_depth}, and possible endings.

    CRITICAL: You MUST generate ALL nodes from depth 0 to depth {story_config.max_depth}.
    - Depth 0: 1 starting node (root)
    - Depth 1: 2 nodes (children of the root node)
    - Depth 2: 4 nodes (each depth 1 node has 2 children)
    - Depth 3: 8 nodes (each depth 2 node has 2 children)
    - Continue this pattern until you reach depth {story_config.max_depth}
    - EVERY non-leaf node (depth 0 to {story_config.max_depth - 1}) MUST have exactly 2 choices and exactly 2 children nodes
    - ONLY nodes at depth {story_config.max_depth} (leaf nodes) should have empty choices and children arrays
    - Total nodes should be approximately {2 ** (story_config.max_depth + 1) - 1} nodes (2^(maxDepth+1) - 1)

    The response MUST be a single, valid JSON object that follows this exact structure:
    {{
      "episode_order": {current_episode_order},
      "title": "Episode Title Here",
      "start_node": {{
        "id": "node_0",
        "text": "The story text for the first scene of the episode...",
        "depth": 0,
        "details": {{
            "npc_emotions": {{"CharacterName": "emotion_state"}},
            "situation": "Brief description of the current situation",
            "relations_update": {{"Character1-Character2": "relationship_change"}}
        }},
        "choices": [
            {{
                "text": "First choice for the player",
                "tags": ["tag1", "tag2"]
            }},
            {{
                "text": "Second choice for the player",
                "tags": ["tag3", "tag4"]
            }}
        ],
        "children": [
            {{
                "id": "node_1",
                "text": "The story continues after the first choice...",
                "depth": 1,
                "details": {{
                    "npc_emotions": {{"CharacterName": "emotion_state"}},
                    "situation": "Brief description",
                    "relations_update": {{}}
                }},
                "choices": [...],
                "children": [...]
            }},
            {{
                "id": "node_2",
                "text": "The story continues after the second choice...",
                "depth": 1,
                "details": {{
                    "npc_emotions": {{"CharacterName": "emotion_state"}},
                    "situation": "Brief description",
                    "relations_update": {{}}
                }},
                "choices": [...],
                "children": [...]
            }}
        ]
      }},
      "endings": [
        {{
          "id": "ep{current_episode_order}_ending_1",
          "title": "Ending Title 1",
          "condition": "cooperative >= 2 AND trusting >= 1",
          "text": "The full ending text describing what happens...",
          "gauge_changes": {{"{story_config.selected_gauge_ids[0] if story_config.selected_gauge_ids else 'gauge1'}": 10, "{story_config.selected_gauge_ids[1] if len(story_config.selected_gauge_ids) > 1 else 'gauge2'}": -5}}
        }},
        {{
          "id": "ep{current_episode_order}_ending_2",
          "title": "Ending Title 2",
          "condition": "aggressive >= 2 OR doubtful >= 2",
          "text": "Another ending text...",
          "gauge_changes": {{"{story_config.selected_gauge_ids[0] if story_config.selected_gauge_ids else 'gauge1'}": -10, "{story_config.selected_gauge_ids[1] if len(story_config.selected_gauge_ids) > 1 else 'gauge2'}": 15}}
        }},
        {{
          "id": "ep{current_episode_order}_ending_3",
          "title": "Ending Title 3",
          "condition": "rational >= 3",
          "text": "A third possible ending...",
          "gauge_changes": {{"{story_config.selected_gauge_ids[0] if story_config.selected_gauge_ids else 'gauge1'}": 5, "{story_config.selected_gauge_ids[1] if len(story_config.selected_gauge_ids) > 1 else 'gauge2'}": 5}}
        }}
      ]
    }}

    IMPORTANT RULES:
    - Each node must have 'id', 'text', 'depth', 'details', 'choices', and 'children'.
    - The 'details' object must include:
      * 'npc_emotions': emotional states of NPCs present in this scene (e.g., {{"Romeo": "passionate", "Juliet": "conflicted"}})
      * 'situation': a brief description of what's happening in this scene
      * 'relations_update': any changes in character relationships (can be empty dict if no changes)
    - The 'children' array of a node should contain the nodes that result from the 'choices' of that same node, in the same order.
    - RECURSIVELY generate children nodes for EVERY node until depth {story_config.max_depth} is reached.
    - Do NOT stop at depth 1 or 2. You MUST continue generating until depth {story_config.max_depth}.
    - Each non-leaf node (depth 0 to {story_config.max_depth - 1}) MUST have exactly 2 choices and exactly 2 children.
    - ONLY nodes at depth {story_config.max_depth} (leaf nodes) should have empty 'choices' and 'children' arrays.
    - Ensure the 'id' of each node is unique within the episode (use node_0, node_1, node_2, etc.).
    - Generate 2-4 possible endings for this episode. Each ending should:
      * Have a unique 'id' field (e.g., "ep{current_episode_order}_ending_1", "ep{current_episode_order}_ending_2", etc.)
      * Have a descriptive title
      * Specify the 'condition' using TAG-BASED LOGIC (e.g., "cooperative >= 2 AND trusting >= 1", "aggressive >= 3 OR doubtful >= 2")
        - Available tags: cooperative, aggressive, cautious, trusting, doubtful, brave, fearful, rational, emotional
        - Use comparison operators: >=, <=, >, <, ==
        - Use logical operators: AND, OR
        - Tags accumulate based on player choices throughout the episode
      * Provide full ending text (200-400 words) describing the outcome
      * Include 'gauge_changes' dict with gauge ID keys and integer values (can be positive or negative)
      * Use the EXACT field name "gauge_changes" (with underscore, not camelCase)

    EXAMPLE FOR maxDepth={story_config.max_depth}:
    - You need {2 ** (story_config.max_depth + 1) - 1} total nodes
    - Depth 0: 1 node → Depth 1: 2 nodes → Depth 2: 4 nodes → Depth 3: 8 nodes (and so on)
    - The deepest nodes (at depth {story_config.max_depth}) are leaf nodes with NO choices or children

    VERIFICATION CHECKLIST BEFORE RESPONDING:
    ✓ Did I generate nodes at ALL depths from 0 to {story_config.max_depth}?
    ✓ Do ALL nodes at depths 0 through {story_config.max_depth - 1} have exactly 2 choices and 2 children?
    ✓ Do ONLY nodes at depth {story_config.max_depth} have empty choices/children arrays?
    ✓ Is the total node count approximately {2 ** (story_config.max_depth + 1) - 1}?
    ✓ Did I generate 2-4 endings with unique 'id' fields (e.g., "ep{current_episode_order}_ending_1")?
    ✓ Do all ending 'condition' fields use TAG-BASED logic (not node paths)?
    ✓ Did I use "gauge_changes" (not "gaugeChanges") in all endings?

    **CRITICAL: All story content (node text, choice text, ending text, titles) MUST be written in Korean (한글).**
    - Only field names and IDs should be in English
    - All narrative content must be in Korean
    """

    # --- Call the LLM and parse the response ---
    print(f"Generating Episode {current_episode_order} with prompt:\n{llm_prompt}")
    response = await director.llm.ainvoke(llm_prompt)
    print(f"🤖 LLM Response content (first 500 chars): {response.content[:500]}")

    generated_episode_data = director._parse_json(response.content)
    print(f"📊 Parsed episode data keys: {generated_episode_data.keys() if isinstance(generated_episode_data, dict) else 'NOT A DICT'}")
    print(f"📊 Full parsed data: {json.dumps(generated_episode_data, ensure_ascii=False, indent=2)[:1000]}")

    # --- Validate and return the Episode object ---
    try:
        # Convert start_node to nodes array if necessary (for backend compatibility)
        if "start_node" in generated_episode_data and "nodes" not in generated_episode_data:
            print("🔄 Converting start_node to nodes array for backend compatibility")
            generated_episode_data["nodes"] = [generated_episode_data["start_node"]]

        new_episode = EpisodeModel(**generated_episode_data)
        print(f"✅ Successfully created EpisodeModel: order={new_episode.episode_order}, title={new_episode.title}")
        print(f"📦 Episode has {len(new_episode.nodes) if new_episode.nodes else 0} nodes")
        return new_episode
    except Exception as e:
        print(f"❌ Error validating episode data: {e}")
        print(f"❌ Episode data that failed: {json.dumps(generated_episode_data, ensure_ascii=False, indent=2)}")
        # Handle validation error, maybe by returning a default error episode
        raise


