import os
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3
import requests
import httpx
import json
from botocore.exceptions import ClientError

from main import main_flow, get_gauges

load_dotenv()

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    region_name=os.getenv('AWS_REGION', 'ap-northeast-2'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)


def download_from_s3(file_key: str, bucket: str = None) -> str:
    """S3에서 파일을 다운로드하여 텍스트로 반환"""
    if bucket is None:
        bucket = os.getenv('AWS_S3_BUCKET', 'story-game-bucket')

    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        return content
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise HTTPException(status_code=404, detail=f"File not found in S3: {file_key}")
        else:
            raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="파일 인코딩 오류. UTF-8 파일을 사용하세요.")


async def upload_to_presigned_url(url: str, data: Dict):
    """미리 서명된 URL로 JSON 데이터를 PUT 요청으로 업로드합니다 (비동기)."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5분 타임아웃
            response = await client.put(
                url,
                content=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()  # 2xx 이외의 상태 코드에 대해 예외 발생
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed with status {e.response.status_code}: {str(e)}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="S3 upload timeout (exceeded 5 minutes)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload result to S3: {str(e)}")


def extract_metadata(story_data: Dict) -> Dict:
    """스토리 데이터에서 메타데이터를 추출합니다."""
    total_nodes = 0
    total_episodes = len(story_data.get("episodes", []))

    # 모든 에피소드의 노드 수 계산
    for episode in story_data.get("episodes", []):
        if "nodes" in episode:
            total_nodes += len(episode["nodes"])

    # 게이지 수 계산
    total_gauges = len(story_data.get("context", {}).get("gauges", []))

    return {
        "total_episodes": total_episodes,
        "total_nodes": total_nodes,
        "total_gauges": total_gauges
    }


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


class EndingConfig(BaseModel):
    """최종 엔딩 타입별 개수 설정"""
    happy: int = 2          # 행복한 엔딩
    tragic: int = 1         # 비극적인 엔딩
    neutral: int = 1        # 중립적인 엔딩
    open: int = 1           # 열린 결말
    bad: int = 0            # 나쁜 엔딩
    bittersweet: int = 0    # 씁쓸한 엔딩


class GenerateRequest(BaseModel):
    novel_text: str
    selected_gauge_ids: List[str]  # 선택한 게이지 ID 2개
    num_episodes: int = 3
    max_depth: int = 3  # 2~5
    ending_config: Optional[EndingConfig] = None  # 엔딩 타입별 개수
    num_episode_endings: int = 3  # 에피소드별 엔딩 개수


class AnalyzeFromS3Request(BaseModel):
    """S3에서 소설 파일을 다운로드하여 분석"""
    file_key: str
    bucket: Optional[str] = "story-game-bucket"
    novel_text: Optional[str] = None  # S3 실패 시 fallback용


class GenerateFromS3Request(BaseModel):
    """S3에서 소설 파일을 다운로드하여 스토리 생성하고 결과를 Pre-signed URL에 업로드"""
    file_key: str
    s3_upload_url: Optional[str] = None  # 결과를 업로드할 Pre-signed URL (Optional)
    s3_file_key: Optional[str] = None  # S3에 저장될 파일 경로
    bucket: Optional[str] = "story-game-bucket"
    selected_gauge_ids: List[str]
    num_episodes: int = 3
    max_depth: int = 3
    ending_config: Optional[EndingConfig] = None
    num_episode_endings: int = 3


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
        # ending_config 변환
        ending_config_dict = None
        if request.ending_config:
            ending_config_dict = {
                "happy": request.ending_config.happy,
                "tragic": request.ending_config.tragic,
                "neutral": request.ending_config.neutral,
                "open": request.ending_config.open,
                "bad": request.ending_config.bad,
                "bittersweet": request.ending_config.bittersweet
            }
            # 0인 항목 제거
            ending_config_dict = {k: v for k, v in ending_config_dict.items() if v > 0}

        result = await main_flow(
            api_key=API_KEY,
            novel_text=request.novel_text,
            selected_gauge_ids=request.selected_gauge_ids,
            num_episodes=request.num_episodes,
            max_depth=request.max_depth,
            ending_config=ending_config_dict,
            num_episode_endings=request.num_episode_endings
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/file")
async def generate_story_from_file(
    file: UploadFile = File(...),
    selected_gauge_ids: str = Form(...),  # 쉼표로 구분된 ID들
    num_episodes: int = Form(3),
    max_depth: int = Form(3),
    ending_config: str = Form("happy:2,tragic:1,neutral:1,open:1"),  # "타입:개수" 형식
    num_episode_endings: int = Form(3)
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

        # ending_config 파싱 ("happy:2,tragic:1" 형식)
        ending_config_dict = {}
        for item in ending_config.split(','):
            if ':' in item:
                etype, count = item.strip().split(':')
                ending_config_dict[etype.strip()] = int(count.strip())

        result = await main_flow(
            api_key=API_KEY,
            novel_text=novel_text,
            selected_gauge_ids=gauge_ids,
            num_episodes=num_episodes,
            max_depth=max_depth,
            ending_config=ending_config_dict if ending_config_dict else None,
            num_episode_endings=num_episode_endings
        )
        return result
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="파일 인코딩 오류. UTF-8 파일을 사용하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-from-s3", response_model=GaugeResponse)
async def analyze_novel_from_s3(request: AnalyzeFromS3Request):
    """
    S3에서 소설 파일을 다운로드하여 분석

    백엔드가 S3에 업로드한 파일을 fileKey로 받아서 분석합니다.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    try:
        # S3에서 파일 다운로드
        novel_text = download_from_s3(request.file_key, request.bucket)

        # 기존 분석 로직 재사용
        result = await get_gauges(API_KEY, novel_text)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/generate-from-s3")
async def generate_story_from_s3(request: GenerateFromS3Request):
    """
    S3에서 소설 파일을 다운로드하여 스토리 생성

    s3_upload_url이 제공되면:
    - 결과를 Pre-signed URL로 S3에 직접 업로드
    - 메타데이터만 반환 (file_key, metadata)

    s3_upload_url이 없으면:
    - 전체 스토리 데이터 반환 (기존 방식)
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
        novel_text = download_from_s3(request.file_key, request.bucket)

        # ending_config 변환
        ending_config_dict = None
        if request.ending_config:
            ending_config_dict = {
                "happy": request.ending_config.happy,
                "tragic": request.ending_config.tragic,
                "neutral": request.ending_config.neutral,
                "open": request.ending_config.open,
                "bad": request.ending_config.bad,
                "bittersweet": request.ending_config.bittersweet
            }
            ending_config_dict = {k: v for k, v in ending_config_dict.items() if v > 0}

        # 기존 생성 로직 재사용
        story_data = await main_flow(
            api_key=API_KEY,
            novel_text=novel_text,
            selected_gauge_ids=request.selected_gauge_ids,
            num_episodes=request.num_episodes,
            max_depth=request.max_depth,
            ending_config=ending_config_dict,
            num_episode_endings=request.num_episode_endings
        )

        # Pre-signed URL이 있으면 S3에 업로드하고 메타데이터만 반환
        if request.s3_upload_url:
            await upload_to_presigned_url(request.s3_upload_url, story_data)

            # 메타데이터 추출
            metadata = extract_metadata(story_data)

            # 메타데이터만 반환 (경량 응답)
            return {
                "status": "success",
                "file_key": request.s3_file_key or "unknown",
                "metadata": metadata
            }
        else:
            # Pre-signed URL이 없으면 전체 데이터 반환 (기존 방식)
            return {
                "status": "success",
                "data": story_data
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")


# ============================================
# 서버 실행
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
