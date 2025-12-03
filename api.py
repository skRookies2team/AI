import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import boto3
import requests
import httpx
import json
from botocore.exceptions import ClientError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from main import main_flow, get_gauges, regenerate_subtree
from storyengine_pkg.generator import generate_single_episode
from storyengine_pkg.models import (
    StoryConfig,
    InitialAnalysis,
    EpisodeModel as Episode,  # Rename to avoid conflict with TypedDict
    GenerateNextEpisodeRequest,
)

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

# Validation Error Handler (422 에러 상세 로깅)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    print("=" * 80)
    print("🚨 422 VALIDATION ERROR 발생!")
    print(f"📍 URL: {request.url}")
    print(f"📍 Method: {request.method}")
    print("=" * 80)
    print("❌ Validation Errors:")
    for error in exc.errors():
        print(f"  - Location: {error['loc']}")
        print(f"    Message: {error['msg']}")
        print(f"    Type: {error['type']}")
        if 'input' in error:
            print(f"    Input: {error['input']}")
        print()
    print("=" * 80)

    # 클라이언트에게도 상세한 에러 반환
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": await request.body() if hasattr(request, 'body') else None
        }
    )

API_KEY = os.environ.get("OPENAI_API_KEY")


# ============================================
# Request/Response 모델
# ============================================

class GaugeInfo(BaseModel):
    id: str
    name: str
    meaning: str
    min_label: str
    max_label: str
    description: Optional[str] = None


class CharacterInfo(BaseModel):
    name: str
    aliases: List[str]
    description: str
    relationships: List[str]


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
    selected_gauges: Optional[List[GaugeInfo]] = None  # 게이지 전체 정보 (옵션)
    num_episodes: int = 3
    max_depth: int = 3  # 2~5
    ending_config: Optional[EndingConfig] = None  # 엔딩 타입별 개수
    num_episode_endings: int = 3  # 에피소드별 엔딩 개수
    file_key: Optional[str] = None  # S3 파일 키 (옵션)
    s3_upload_url: Optional[str] = None  # S3 업로드 Pre-signed URL (옵션)


class AnalyzeFromS3Request(BaseModel):
    """S3에서 소설 파일을 다운로드하여 분석"""
    file_key: str
    bucket: Optional[str] = "story-game-bucket"
    s3_upload_url: Optional[str] = None  # 결과를 업로드할 Pre-signed URL
    result_file_key: Optional[str] = None  # 반환할 파일 키
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


class ParentNodeInfo(BaseModel):
    """재생성할 부모 노드 정보"""
    nodeId: str
    text: str
    choices: List[str]
    situation: Optional[str] = None
    npcEmotions: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    depth: int


class SubtreeRegenerationRequest(BaseModel):
    """서브트리 재생성 요청"""
    episodeTitle: str
    episodeOrder: int
    parentNode: ParentNodeInfo
    currentDepth: int
    maxDepth: int
    novelContext: str
    previousChoices: List[str] = []
    selectedGaugeIds: List[str]

    # 캐싱된 분석 결과 (성능 최적화)
    summary: Optional[str] = None
    charactersJson: Optional[str] = None
    gaugesJson: Optional[str] = None


class RegeneratedNode(BaseModel):
    """재생성된 노드"""
    id: str
    text: str
    choices: List[str]
    depth: int
    details: Optional[Dict] = None
    children: List[Any] = []


class SubtreeRegenerationResponse(BaseModel):
    """서브트리 재생성 응답"""
    status: str
    message: str
    regeneratedNodes: List[Dict]
    totalNodesRegenerated: int


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
    스토리 생성 - relay-server에서 호출

    novelText를 받아서 스토리를 생성하고, s3_upload_url이 있으면 S3에 업로드
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

        print(f"🎬 스토리 생성 시작 (에피소드: {request.num_episodes}, 깊이: {request.max_depth})")
        story_data = await main_flow(
            api_key=API_KEY,
            novel_text=request.novel_text,
            selected_gauge_ids=request.selected_gauge_ids,
            num_episodes=request.num_episodes,
            max_depth=request.max_depth,
            ending_config=ending_config_dict,
            num_episode_endings=request.num_episode_endings
        )
        print(f"✅ 스토리 생성 완료")

        # Pre-signed URL이 있으면 S3에 업로드하고 메타데이터만 반환
        if request.s3_upload_url:
            print(f"📤 S3에 업로드 시작")
            await upload_to_presigned_url(request.s3_upload_url, story_data)
            print(f"✅ S3 업로드 완료")

            # 메타데이터 추출
            metadata = extract_metadata(story_data)

            # 메타데이터만 반환 (경량 응답)
            return {
                "status": "success",
                "file_key": request.file_key or "unknown",
                "data": {
                    "metadata": metadata
                }
            }
        else:
            # Pre-signed URL이 없으면 전체 데이터 반환 (기존 방식)
            return {
                "status": "success",
                "data": story_data
            }

    except Exception as e:
        import traceback
        error_detail = f"Story generation failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ 오류 발생:\n{error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/generate-next-episode", response_model=Episode)
async def generate_next_episode_endpoint(request: GenerateNextEpisodeRequest):
    """
    Generates a single episode sequentially.
    """
    print("=" * 60)
    print("📥 /generate-next-episode 요청 수신")
    print(f"  - Current Episode Order: {request.current_episode_order}")
    print(f"  - Story Config: {request.story_config}")
    print(f"  - Has Previous Episode: {request.previous_episode is not None}")
    print("=" * 60)

    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    try:
        newly_generated_episode = await generate_single_episode(
            api_key=API_KEY,
            initial_analysis=request.initial_analysis,
            story_config=request.story_config,
            novel_context=request.novel_context,
            current_episode_order=request.current_episode_order,
            previous_episode_data=request.previous_episode
        )
        return newly_generated_episode
    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
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


@app.post("/analyze-from-s3")
async def analyze_novel_from_s3(request: AnalyzeFromS3Request):
    """
    S3에서 소설 파일을 다운로드하여 분석

    백엔드가 S3에 업로드한 파일을 fileKey로 받아서 분석합니다.

    s3_upload_url이 제공되면:
    - 분석 결과를 Pre-signed URL로 S3에 업로드
    - fileKey만 반환

    s3_upload_url이 없으면:
    - 전체 분석 결과 반환 (기존 방식)
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    try:
        # 1. S3에서 소설 다운로드
        novel_text = download_from_s3(request.file_key, request.bucket)

        # 2. 분석 (요약, 캐릭터, 게이지)
        result = await get_gauges(API_KEY, novel_text)

        # 3. Pre-signed URL이 있으면 S3에 업로드하고 fileKey만 반환
        if request.s3_upload_url:
            # 4. S3에 업로드 (Pre-signed URL 사용)
            await upload_to_presigned_url(request.s3_upload_url, result)

            # 5. fileKey만 반환
            return {"file_key": request.result_file_key or request.file_key}

        # Pre-signed URL이 없으면 전체 결과 반환 (기존 방식)
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
        print(f"📥 S3에서 파일 다운로드 시작: {request.file_key}")
        novel_text = download_from_s3(request.file_key, request.bucket)
        print(f"✅ 다운로드 완료 (텍스트 길이: {len(novel_text)}자)")

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
        print(f"🎬 스토리 생성 시작 (에피소드: {request.num_episodes}, 깊이: {request.max_depth})")
        story_data = await main_flow(
            api_key=API_KEY,
            novel_text=novel_text,
            selected_gauge_ids=request.selected_gauge_ids,
            num_episodes=request.num_episodes,
            max_depth=request.max_depth,
            ending_config=ending_config_dict,
            num_episode_endings=request.num_episode_endings
        )
        print(f"✅ 스토리 생성 완료")

        # Pre-signed URL이 있으면 S3에 업로드하고 메타데이터만 반환
        if request.s3_upload_url:
            print(f"📤 S3에 업로드 시작")
            await upload_to_presigned_url(request.s3_upload_url, story_data)
            print(f"✅ S3 업로드 완료")

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
        import traceback
        error_detail = f"Story generation failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ 오류 발생:\n{error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/regenerate-subtree", response_model=SubtreeRegenerationResponse)
async def regenerate_node_subtree(request: SubtreeRegenerationRequest):
    """
    수정된 부모 노드를 기반으로 하위 서브트리를 재생성합니다.

    Top-Down 방식으로 상위 노드 수정 시 하위 노드들을 새로운 내용에 맞춰 재생성합니다.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    try:
        print(f"🔄 서브트리 재생성 요청 받음")
        print(f"  에피소드: {request.episodeTitle} (#{request.episodeOrder})")
        print(f"  부모 노드: {request.parentNode.nodeId} (depth {request.currentDepth}/{request.maxDepth})")

        # 부모 노드 정보를 Dict로 변환
        parent_node_dict = {
            "nodeId": request.parentNode.nodeId,
            "text": request.parentNode.text,
            "choices": request.parentNode.choices,
            "situation": request.parentNode.situation,
            "npcEmotions": request.parentNode.npcEmotions,
            "tags": request.parentNode.tags,
            "depth": request.parentNode.depth
        }

        # 서브트리 재생성 실행 (캐시된 정보 활용)
        result = await regenerate_subtree(
            api_key=API_KEY,
            parent_node=parent_node_dict,
            novel_context=request.novelContext,
            selected_gauge_ids=request.selectedGaugeIds,
            current_depth=request.currentDepth,
            max_depth=request.maxDepth,
            episode_title=request.episodeTitle,
            previous_choices=request.previousChoices,
            # 캐싱된 정보 전달 (새로 분석 건너뛰기)
            cached_summary=request.summary,
            cached_characters_json=request.charactersJson,
            cached_gauges_json=request.gaugesJson
        )

        print(f"✅ 서브트리 재생성 완료: {result['totalNodesRegenerated']}개 노드")

        return SubtreeRegenerationResponse(
            status=result["status"],
            message=result["message"],
            regeneratedNodes=result["regeneratedNodes"],
            totalNodesRegenerated=result["totalNodesRegenerated"]
        )

    except Exception as e:
        import traceback
        error_detail = f"Subtree regeneration failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ 서브트리 재생성 오류:\n{error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for relay server
    """
    return {
        "status": "healthy",
        "service": "AI Story Generation Server",
        "version": "1.0.0"
    }


# ============================================
# 서버 실행
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
