# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered interactive story generation engine that transforms novel text into branching narrative experiences. The system is part of the "IF STORY" service, working as the AI server component in a larger microservices architecture:

- **AI Server (this repo)**: Generates interactive story content using LLMs
- **Backend Server** (`../story-backend`): Main application server with MariaDB
- **Relay Server** (`../relay-server`): Coordinates AI generation and image processing
- **Frontend** (`../Front`): User interface

## Build & Run Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**FastAPI Server (production mode):**
```bash
uvicorn api:app --reload
# Server runs at http://127.0.0.1:8000
# API docs at http://127.0.0.1:8000/docs
```

**Command-line Mode (testing):**
```bash
python main.py
```

### Environment Variables

Create a `.env` file with:
```
OPENAI_API_KEY="your-api-key-here"
AWS_REGION="ap-northeast-2"
AWS_ACCESS_KEY_ID="your-key"
AWS_SECRET_ACCESS_KEY="your-secret"
AWS_S3_BUCKET="story-game-bucket"
```

## Architecture

### Core Workflow

The system follows a **sequential episode generation** model where episodes are created one at a time, allowing user interaction between episodes:

1. **Initial Analysis Phase** (first request only):
   - Novel summary generation
   - Character extraction with detailed relationships and citations
   - Gauge system design (metrics that track story state)
   - Final endings design (based on accumulated gauge values)

2. **Episode Generation Phase** (repeatable):
   - Episode splitting based on novel structure
   - Branching narrative tree construction using LangGraph
   - Story node creation with choices and gauge impacts
   - Episode-specific endings generation

3. **Story Flow**:
   - Each episode contains a tree of nodes (depth configurable 2-5)
   - Each node has choices that affect gauge values
   - Episode endings determined by accumulated tag scores
   - Final ending determined by total gauge accumulation across all episodes

### Key Components

**`storyengine_pkg/` Package Structure:**
- `director.py`: `InteractiveStoryDirector` - Main orchestration class using LangGraph state machines
- `generator.py`: Episode and node generation logic using OpenAI LLM
- `models.py`: TypedDict and Pydantic model definitions
- `utils.py`: Helper functions for calculations, file I/O, node traversal
- `validation.py`: Story tree validation (dead ends, gauge balance, reachability)
- `crud.py`: Node/choice editing operations
- `simulation.py`: Playthrough simulation and ending verification
- `export.py`: Export to Markdown/HTML/JSON formats

**Entry Points:**
- `api.py`: FastAPI server with endpoints for story generation
- `main.py`: Command-line interface for local testing

### LangGraph State Machine

The story generation uses LangGraph to model the workflow as a state graph. Key states:
- Episode splitting
- Node expansion (parallel processing of tree levels)
- Ending generation
- Result aggregation

The graph uses `Send()` for dynamic parallel processing when expanding multiple nodes at the same depth.

### Data Flow Pattern

**Sequential API Pattern:**
```
Backend → AI Server: POST /generate-next-episode
  Request: {
    initial_analysis,
    story_config,
    novel_context,
    current_episode_order,
    previous_episode (optional)
  }
  Response: Single Episode object

Backend → Relay Server → Image AI → S3
  (Image generation happens in relay server, not here)
```

## Important Patterns

### Async/Await Usage
All LLM calls and I/O operations use Python `asyncio`. The director methods are `async` and should be `await`ed.

### Citation System
Character extraction includes `[cite: line_number]` references to ground AI outputs in source text. This prevents hallucination.

### Gauge System
- Gauges are numeric metrics (0-100) representing story state
- Player choices have tags that map to gauge changes
- Final endings are conditional on gauge thresholds
- Episode endings are conditional on tag accumulation

### Node Tree Structure
- Nodes have `depth`, `parent_id`, and `node_type` fields
- `node_type`: "normal" | "climax" | "ending"
- Climax nodes appear at max_depth - 1
- Ending nodes are terminal (no choices)

### S3 Integration
The AI server uses S3 for:
- Downloading source novel text via `download_from_s3()`
- Uploading generated story JSON via presigned URLs
- Uses `httpx` for async HTTP operations

## Common Development Tasks

### Modifying LLM Prompts
Prompts are embedded in `director.py` methods. Each generation step (character extraction, gauge design, node creation) has detailed prompt engineering. The prompts include:
- JSON schema requirements
- Citation formatting (`[cite: N]`)
- Korean language instructions
- Context from previous episodes

### Adjusting Tree Depth Logic
- Climax nodes: `depth == max_depth - 1`
- Ending nodes: `depth == max_depth`
- Normal nodes: `depth < max_depth - 1`

This logic is in `generator.py:generate_single_episode()` and must stay consistent.

### Adding New Endpoints
1. Define Pydantic request/response models in `api.py`
2. Implement handler using `async def`
3. Call director methods from `main.py` or `generator.py`
4. Handle S3 operations with error handling
5. Return JSON responses matching backend DTOs

### Testing Story Generation
Use `simulation.py` functions:
```python
from storyengine_pkg import simulate_playthrough, get_all_possible_endings

# Test a random playthrough
result = simulate_playthrough(episode_data)

# Verify all endings are reachable
endings = get_all_possible_endings(episode_data)
```

## API Endpoints

### `POST /generate`
Legacy endpoint - generates entire multi-episode story at once. Use for initial testing only.

### `POST /generate-next-episode`
**Primary endpoint** - generates one episode sequentially.
- First episode: Uses `initial_analysis` from request
- Subsequent episodes: Uses `previous_episode` state as context

### `GET /health`
Health check endpoint.

## Korean Language Requirement

All AI-generated content must be in Korean. This is enforced in prompts with "대답은 무조건 한글로 할 것" and similar instructions. Do not remove these directives.

## Dependencies

Key libraries:
- `langchain-openai`: LLM interface
- `langgraph`: State graph orchestration
- `fastapi` + `uvicorn`: Web server
- `boto3`: AWS S3 integration
- `pydantic`: Data validation
- `httpx`: Async HTTP client

## Files to Know

- `episode_story.json`: Example output from story generation
- `send_request.py`: Test client for API endpoints
- `test_s3_connection.py`: S3 connectivity test
- `AI_REFACTOR_GUIDE.md`: Detailed guide for sequential generation refactoring
- `SYSTEM_FLOW.md`: System architecture diagrams (Mermaid)
- `상세플로우.md`: Detailed Korean documentation
- `GEMINI.md`: Gemini AI integration notes (if applicable)
