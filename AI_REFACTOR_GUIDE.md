# AI Server Refactoring Guide: Sequential Episode Generation

## 1. Objective

To modify the AI server's story generation logic from the current "all-in-one" model to a "step-by-step, sequential" model. This will enable the backend to request one episode at a time, allowing for user interaction and editing between episodes.

## 2. Summary of Changes

1.  **Deprecate Endpoint:** The existing `POST /generate` endpoint, which creates an entire multi-episode story at once, will be deprecated or removed.
2.  **New Endpoint:** A new endpoint, `POST /generate-next-episode`, will be created. This endpoint is responsible for generating only a single episode based on the context provided.
3.  **Refactor Core Logic:** The core generation engine must be adapted to generate a single episode, using the previous episode's outcome as the starting context for the next one.

---

## 3. New API Specification

### `POST /generate-next-episode`

**Description:** Generates a single episode. If it's the first episode, it uses the initial analysis. If it's a subsequent episode, it uses the state and ending of the previous episode as context.

#### Request Body

The request body should be a JSON object with the following structure. Here is an example using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class StoryConfig(BaseModel):
    num_episodes: int
    max_depth: int
    selected_gauge_ids: List[str]
    # Add other config fields if necessary

class InitialAnalysis(BaseModel):
    summary: str
    characters: List[Dict] # or a more specific Character model
    # Add other analysis fields if necessary

# This model should match the 'EpisodeDto' from the Java backend
class Episode(BaseModel):
    episode_order: int
    title: str
    start_node: Dict # or a more specific StoryNode model
    # Include all necessary fields for a complete episode

class GenerateNextEpisodeRequest(BaseModel):
    initial_analysis: InitialAnalysis
    story_config: StoryConfig
    novel_context: str
    current_episode_order: int = Field(..., description="The order of the episode to be generated, e.g., 1, 2, 3...")
    previous_episode: Optional[Episode] = Field(None, description="The full data of the previous episode. Null if generating the first episode.")
```

#### Response Body

The response should be the JSON object of the newly created single episode.

```python
# The response body is simply the 'Episode' model
# (Matches 'EpisodeDto' in the Java backend)

class Episode(BaseModel):
    episode_order: int
    title: str
    start_node: Dict # or a StoryNode model
    # ...
```

---

## 4. Implementation Guide

This guide assumes you are using FastAPI.

### 4.1. `api.py` (or `main.py`) Modifications

Add the Pydantic models described above and the new endpoint.

```python
# In api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Assuming your app is initialized here
app = FastAPI()

# -------------------------------------------------------------------
# 1. DEFINE PYDANTIC MODELS (as specified in Section 3)
# -------------------------------------------------------------------

class StoryConfig(BaseModel):
    num_episodes: int
    max_depth: int
    selected_gauge_ids: List[str]

class InitialAnalysis(BaseModel):
    summary: str
    characters: List[Dict]

class Episode(BaseModel):
    episode_order: int
    title: str
    start_node: Dict # Using Dict for simplicity, can be a detailed StoryNode model

class GenerateNextEpisodeRequest(BaseModel):
    initial_analysis: InitialAnalysis
    story_config: StoryConfig
    novel_context: str
    current_episode_order: int = Field(..., description="The order of the episode to be generated, e.g., 1, 2, 3...")
    previous_episode: Optional[Episode] = Field(None, description="The full data of the previous episode. Null if generating the first episode.")


# -------------------------------------------------------------------
# 2. COMMENT OUT OR REMOVE THE OLD /generate ENDPOINT
# -------------------------------------------------------------------

# @app.post("/generate")
# async def generate_full_story(request: ...):
#     """
#     DEPRECATED: This endpoint generates all episodes at once.
#     It will be replaced by the sequential generation flow.
#     """
#     # Old logic here...
#     pass


# -------------------------------------------------------------------
# 3. ADD THE NEW /generate-next-episode ENDPOINT
# -------------------------------------------------------------------

# Import your core logic function
# from storyengine_pkg.generator import generate_single_episode

@app.post("/generate-next-episode", response_model=Episode)
async def generate_next_episode_endpoint(request: GenerateNextEpisodeRequest):
    """
    Generates a single episode sequentially.
    """
    try:
        # This function will contain your core LLM logic
        newly_generated_episode = generate_single_episode(
            initial_analysis=request.initial_analysis,
            story_config=request.story_config,
            novel_context=request.novel_context,
            current_episode_order=request.current_episode_order,
            previous_episode_data=request.previous_episode
        )
        return newly_generated_episode
    except Exception as e:
        # Add more specific error handling
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# 4. (Example) CORE LOGIC FUNCTION
# This function would likely live in a separate file, e.g., storyengine_pkg/generator.py
# -------------------------------------------------------------------

def generate_single_episode(initial_analysis: InitialAnalysis,
                            story_config: StoryConfig,
                            novel_context: str,
                            current_episode_order: int,
                            previous_episode_data: Optional[Episode]) -> Episode:
    """
    Contains the core logic to generate one episode.
    """
    
    # --- Determine the context for the LLM prompt ---
    if previous_episode_data is None:
        # This is the first episode.
        context_prompt = f"""
        Based on the initial analysis of a novel (summary: {initial_analysis.summary}), 
        create the very first episode of an interactive story.
        The story should have {story_config.num_episodes} episodes in total.
        The selected themes (gauges) are: {', '.join(story_config.selected_gauge_ids)}.
        This is episode 1.
        """
    else:
        # This is a subsequent episode (e.g., Ep 2, 3...).
        # You need to analyze the previous episode to create a smooth transition.
        # For example, find the possible endings of the previous episode.
        
        # This is a simplified example. You might need a more robust way
        # to summarize the outcome of the previous episode.
        previous_episode_summary = f"The previous episode, '{previous_episode_data.title}', ended with the protagonist facing a choice that led them to..."

        context_prompt = f"""
        The story continues. The previous episode ended like this: {previous_episode_summary}.
        Now, create episode {current_episode_order} of the story.
        Continue to incorporate the main themes: {', '.join(story_config.selected_gauge_ids)}.
        """

    # --- Construct the main LLM prompt ---
    # This prompt asks the LLM to generate the content for one full episode
    llm_prompt = f"""
    {context_prompt}

    Generate the content for this episode, including a title, a starting node,
    and a branching narrative tree up to a depth of {story_config.max_depth}.

    Return the result as a single JSON object with the following structure:
    {{
      "episode_order": {current_episode_order},
      "title": "Episode Title Here",
      "start_node": {{ ... (full story node structure) ... }}
    }}
    """

    # --- Call the LLM and parse the response ---
    # llm_response_json = call_your_llm(llm_prompt)
    # generated_episode_data = json.loads(llm_response_json)

    # This is placeholder data for demonstration
    generated_episode_data = {
        "episode_order": current_episode_order,
        "title": f"Episode {current_episode_order}: A New Beginning",
        "start_node": {
            "id": "ep{}_node_0".format(current_episode_order),
            "text": "This is the starting text for the new episode.",
            "choices": ["Choice 1", "Choice 2"],
            "children": [] # The LLM would generate this recursively
        }
    }


    # --- Create and return the Episode object ---
    new_episode = Episode(**generated_episode_data)
    
    return new_episode

```

## 5. Backend Call Example

For reference, here is how the Java backend will now call this new endpoint:

```java
// In StoryManagementService.java

// ... webClient setup ...

// When generating the NEXT episode
GenerateNextEpisodeRequest requestBody = new GenerateNextEpisodeRequest(
    initialAnalysis,
    storyConfig,
    novelContext,
    currentEpisodeOrder, // e.g., 2
    previousEpisode      // The EpisodeDto object for episode 1
);

EpisodeDto newEpisode = webClient.post()
    .uri("/generate-next-episode")
    .bodyValue(requestBody)
    .retrieve()
    .bodyToMono(EpisodeDto.class)
    .block();

// The backend will then save this newEpisode.
```
