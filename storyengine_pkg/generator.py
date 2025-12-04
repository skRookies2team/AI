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
        context_prompt = f"""
        Based on the following novel, create the very first episode of an interactive story.

        ORIGINAL NOVEL CONTEXT:
        {novel_context[:3000]}...

        Summary: {initial_analysis.summary if initial_analysis.summary else "Use the novel context above"}
        Main characters: {json.dumps(initial_analysis.characters, ensure_ascii=False, indent=2)}

        The story should have {story_config.num_episodes} episodes in total.
        The selected themes (gauges) are: {', '.join(story_config.selected_gauge_ids)}.
        This is episode 1.

        IMPORTANT: Stay faithful to the original novel's setting, characters, and plot.
        Transform the novel into an interactive experience where players make choices that affect the story.
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
          "title": "Ending Title 1",
          "condition": "Condition to reach this ending (e.g., 'Choose path A at node_5')",
          "text": "The full ending text describing what happens...",
          "gauge_changes": {{"{story_config.selected_gauge_ids[0] if story_config.selected_gauge_ids else 'gauge1'}": 10, "{story_config.selected_gauge_ids[1] if len(story_config.selected_gauge_ids) > 1 else 'gauge2'}": -5}}
        }},
        {{
          "title": "Ending Title 2",
          "condition": "Condition to reach this ending (e.g., 'Choose path B at node_5')",
          "text": "Another ending text...",
          "gauge_changes": {{"{story_config.selected_gauge_ids[0] if story_config.selected_gauge_ids else 'gauge1'}": -10, "{story_config.selected_gauge_ids[1] if len(story_config.selected_gauge_ids) > 1 else 'gauge2'}": 15}}
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
      * Have a descriptive title
      * Specify the condition/path to reach it (which leaf nodes at depth {story_config.max_depth} lead to this ending)
      * Provide full ending text (200-400 words)
      * Include gauge changes that reflect the ending's outcome (positive/negative values for the selected gauges)
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
    ✓ Did I use "gauge_changes" (not "gaugeChanges") in all endings?
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


