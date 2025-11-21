import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from main import main_flow, get_gauges

load_dotenv()

app = FastAPI(
    title="Interactive Story Engine API",
    description="소설 텍스트를 인터랙티브 스토리로 변환하는 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("OPENAI_API_KEY")


# ============================================
# Request/Response 모델
# ============================================

class GaugeRequest(BaseModel):
    novel_text: str


class GenerateRequest(BaseModel):
    novel_text: str
    selected_gauge_ids: List[str]  # 선택한 게이지 ID 2개
    num_episodes: int = 3
    max_depth: int = 3  # 2~5


class GaugeInfo(BaseModel):
    id: str
    name: str
    meaning: str
    min_label: str
    max_label: str
    description: str


class CharacterInfo(BaseModel):
    name: str
    aliases: List[str]
    description: str
    relationships: List[str]


class GaugeResponse(BaseModel):
    summary: str
    characters: List[dict]
    gauges: List[dict]


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"status": "ok", "message": "Interactive Story Engine API"}


@app.post("/analyze", response_model=GaugeResponse)
async def analyze_novel(request: GaugeRequest):
    """
    소설 분석 - 요약, 캐릭터, 게이지 제안 반환

    프론트엔드에서 게이지 선택 UI를 위해 먼저 호출
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    try:
        result = await get_gauges(API_KEY, request.novel_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file", response_model=GaugeResponse)
async def analyze_novel_file(file: UploadFile = File(...)):
    """
    소설 파일 분석 - txt 파일 업로드
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="txt 파일만 지원합니다.")

    try:
        content = await file.read()
        novel_text = content.decode('utf-8')
        result = await get_gauges(API_KEY, novel_text)
        return result
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="파일 인코딩 오류. UTF-8 파일을 사용하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_story(request: GenerateRequest):
    """
    인터랙티브 스토리 생성

    Parameters:
    - novel_text: 소설 텍스트
    - selected_gauge_ids: 선택한 게이지 ID 리스트 (2개)
    - num_episodes: 에피소드 개수 (기본값: 3)
    - max_depth: 트리 깊이 (기본값: 3, 범위: 2~5)
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    # 유효성 검사
    if len(request.selected_gauge_ids) < 2:
        raise HTTPException(status_code=400, detail="게이지 ID를 2개 이상 선택해야 합니다.")

    if not (2 <= request.max_depth <= 5):
        raise HTTPException(status_code=400, detail="트리 깊이는 2~5 사이여야 합니다.")

    if request.num_episodes < 1:
        raise HTTPException(status_code=400, detail="에피소드 개수는 1 이상이어야 합니다.")

    try:
        result = await main_flow(
            api_key=API_KEY,
            novel_text=request.novel_text,
            selected_gauge_ids=request.selected_gauge_ids,
            num_episodes=request.num_episodes,
            max_depth=request.max_depth
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/file")
async def generate_story_from_file(
    file: UploadFile = File(...),
    selected_gauge_ids: str = Form(...),  # 쉼표로 구분된 ID들
    num_episodes: int = Form(3),
    max_depth: int = Form(3)
):
    """
    파일 업로드로 인터랙티브 스토리 생성

    Parameters:
    - file: txt 파일
    - selected_gauge_ids: 쉼표로 구분된 게이지 ID (예: "hope,trust")
    - num_episodes: 에피소드 개수
    - max_depth: 트리 깊이 (2~5)
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="txt 파일만 지원합니다.")

    # 게이지 ID 파싱
    gauge_ids = [g.strip() for g in selected_gauge_ids.split(',') if g.strip()]
    if len(gauge_ids) < 2:
        raise HTTPException(status_code=400, detail="게이지 ID를 2개 이상 선택해야 합니다.")

    if not (2 <= max_depth <= 5):
        raise HTTPException(status_code=400, detail="트리 깊이는 2~5 사이여야 합니다.")

    try:
        content = await file.read()
        novel_text = content.decode('utf-8')

        result = await main_flow(
            api_key=API_KEY,
            novel_text=novel_text,
            selected_gauge_ids=gauge_ids,
            num_episodes=num_episodes,
            max_depth=max_depth
        )
        return result
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="파일 인코딩 오류. UTF-8 파일을 사용하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 서버 실행
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
