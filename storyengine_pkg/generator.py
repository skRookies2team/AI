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
        Based on the initial analysis of a novel (summary: {initial_analysis.summary}), 
        create the very first episode of an interactive story.
        The story should have {story_config.num_episodes} episodes in total.
        The selected themes (gauges) are: {', '.join(story_config.selected_gauge_ids)}.
        This is episode 1.
        The main characters are: {json.dumps(initial_analysis.characters, ensure_ascii=False, indent=2)}
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
    and a branching narrative tree up to a depth of {story_config.max_depth}.

    The response MUST be a single, valid JSON object that follows this exact structure:
    {{
      "episode_order": {current_episode_order},
      "title": "Episode Title Here",
      "start_node": {{
        "id": "node_0",
        "text": "The story text for the first scene of the episode...",
        "depth": 0,
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
                "choices": [...],
                "children": [...]
            }},
            {{
                "id": "node_2",
                "text": "The story continues after the second choice...",
                "depth": 1,
                "choices": [...],
                "children": [...]
            }}
        ]
      }}
    }}

    - Each node must have 'id', 'text', 'depth', 'choices', and 'children'.
    - The 'children' array of a node should contain the nodes that result from the 'choices' of that same node, in the same order.
    - Recursively generate children nodes until the max_depth of {story_config.max_depth} is reached.
    - Nodes at max_depth should have an empty 'choices' and 'children' array.
    - Ensure the 'id' of each node is unique within the episode.
    """

    # --- Call the LLM and parse the response ---
    print(f"Generating Episode {current_episode_order} with prompt:\n{llm_prompt}")
    response = await director.llm.ainvoke(llm_prompt)
    generated_episode_data = director._parse_json(response.content)

    # --- Validate and return the Episode object ---
    try:
        new_episode = EpisodeModel(**generated_episode_data)
        return new_episode
    except Exception as e:
        print(f"Error validating episode data: {e}")
        # Handle validation error, maybe by returning a default error episode
        raise


