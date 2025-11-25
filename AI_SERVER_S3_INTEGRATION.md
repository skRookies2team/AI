# AI 서버 S3 통합 작업 가이드

## 배경

현재 백엔드가 S3에서 파일을 다운로드해서 AI 서버로 전송하는 비효율적인 구조를 개선하기 위해,
**AI 서버가 S3에서 직접 파일을 다운로드**하도록 변경합니다.

## 개선 전/후 비교

### 현재 (비효율적)
```
프론트 → S3 (파일 업로드)
프론트 → 백엔드 (fileKey)
백엔드 → S3 (파일 다운로드) ❌ 메모리 부담
백엔드 → AI 서버 (텍스트 전송) ❌ 네트워크 부담
```

### 개선 후 (효율적)
```
프론트 → S3 (파일 업로드)
프론트 → 백엔드 (fileKey)
백엔드 → AI 서버 (fileKey만 전달) ✅
AI 서버 → S3 (직접 다운로드) ✅
```

---

## AI 서버 작업 내용

### 1. 환경 설정

#### requirements.txt에 추가
```txt
boto3==1.34.0
```

#### 환경 변수 설정
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="ap-northeast-2"
export AWS_S3_BUCKET="story-game-bucket"
```

### 2. S3 클라이언트 초기화

```python
# main.py 또는 app.py 상단에 추가

import boto3
import os
from botocore.exceptions import ClientError

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
```

### 3. 새 엔드포인트 추가

#### 3-1. 소설 분석 (S3 방식)

```python
from pydantic import BaseModel
from typing import Optional

class AnalyzeFromS3Request(BaseModel):
    file_key: str
    bucket: Optional[str] = "story-game-bucket"
    novel_text: Optional[str] = None  # S3 실패 시 fallback용

@app.post("/analyze-from-s3")
async def analyze_novel_from_s3(request: AnalyzeFromS3Request):
    """
    S3에서 소설 파일을 다운로드하여 분석

    Request:
    {
        "file_key": "uploads/abc123_novel.txt",
        "bucket": "story-game-bucket"
    }

    Response:
    {
        "summary": "소설 요약 500자...",
        "characters": [...],
        "gauges": [...]
    }
    """
    try:
        # S3에서 파일 다운로드
        novel_text = download_from_s3(request.file_key, request.bucket)

        # 기존 분석 로직 재사용
        summary = await analyze_summary(novel_text)
        characters = await extract_characters(novel_text)
        gauges = await suggest_gauges(novel_text)

        return {
            "summary": summary,
            "characters": characters,
            "gauges": gauges
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
```

#### 3-2. 스토리 생성 (S3 방식)

```python
class GenerateFromS3Request(BaseModel):
    file_key: Optional[str] = None
    bucket: Optional[str] = "story-game-bucket"
    novel_text: Optional[str] = None  # S3가 아닌 경우 사용
    selected_gauge_ids: list[str]
    num_episodes: int
    max_depth: int
    ending_config: dict
    num_episode_endings: int

@app.post("/generate-from-s3")
async def generate_story_from_s3(request: GenerateFromS3Request):
    """
    S3에서 소설 파일을 다운로드하여 스토리 생성

    Request:
    {
        "file_key": "uploads/abc123_novel.txt",
        "bucket": "story-game-bucket",
        "selected_gauge_ids": ["civilization", "unity"],
        "num_episodes": 3,
        "max_depth": 3,
        "ending_config": {
            "happy": 2,
            "tragic": 1,
            "neutral": 1,
            "open": 1,
            "bad": 0
        },
        "num_episode_endings": 3
    }

    Response:
    {
        "status": "success",
        "message": "Story generated successfully",
        "data": {
            "metadata": {...},
            "context": {...},
            "episodes": [...]
        }
    }
    """
    try:
        # S3에서 파일 다운로드 (file_key가 있으면 S3, 없으면 novel_text 사용)
        if request.file_key:
            novel_text = download_from_s3(request.file_key, request.bucket)
        elif request.novel_text:
            novel_text = request.novel_text
        else:
            raise HTTPException(status_code=400, detail="Either file_key or novel_text is required")

        # 기존 생성 로직 재사용
        story_data = await generate_story(
            novel_text=novel_text,
            selected_gauge_ids=request.selected_gauge_ids,
            num_episodes=request.num_episodes,
            max_depth=request.max_depth,
            ending_config=request.ending_config,
            num_episode_endings=request.num_episode_endings
        )

        return {
            "status": "success",
            "message": "Story generated successfully",
            "data": story_data
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None
        }
```

### 4. 기존 엔드포인트 유지

**중요: 기존 `/analyze`와 `/generate` 엔드포인트는 그대로 유지**하세요.
하위 호환성을 위해 두 가지 방식 모두 지원합니다:

- `/analyze` - 기존 방식 (텍스트 직접 전송)
- `/analyze-from-s3` - 새 방식 (S3 fileKey 전송)
- `/generate` - 기존 방식
- `/generate-from-s3` - 새 방식

---

## 테스트 방법

### 1. S3 연결 테스트
```python
# test_s3.py
import boto3

s3_client = boto3.client('s3', region_name='ap-northeast-2')

# 버킷 목록 확인
response = s3_client.list_buckets()
print("Buckets:", [b['Name'] for b in response['Buckets']])

# 테스트 파일 다운로드
response = s3_client.get_object(
    Bucket='story-game-bucket',
    Key='uploads/test.txt'
)
content = response['Body'].read().decode('utf-8')
print("Content:", content[:100])
```

### 2. 엔드포인트 테스트
```bash
# 분석 테스트
curl -X POST http://localhost:8000/analyze-from-s3 \
  -H "Content-Type: application/json" \
  -d '{
    "file_key": "uploads/test_novel.txt",
    "bucket": "story-game-bucket"
  }'

# 생성 테스트
curl -X POST http://localhost:8000/generate-from-s3 \
  -H "Content-Type: application/json" \
  -d '{
    "file_key": "uploads/test_novel.txt",
    "bucket": "story-game-bucket",
    "selected_gauge_ids": ["civilization", "unity"],
    "num_episodes": 3,
    "max_depth": 3,
    "ending_config": {"happy": 2, "tragic": 1},
    "num_episode_endings": 3
  }'
```

---

## 백엔드 변경 완료 사항

백엔드는 이미 준비 완료되었습니다:

### 변경된 파일
1. **NovelAnalysisRequestDto.java**
   - `fileKey`, `bucket` 필드 추가
   - `isS3Mode()` 메서드 추가

2. **StoryGenerationRequestDto.java**
   - `fileKey`, `bucket` 필드 추가

3. **StoryCreation.java**
   - `s3FileKey` 필드 추가

4. **StoryManagementService.java**
   - `uploadNovelFromS3()` 메서드 수정 - fileKey만 저장
   - `startAnalysisFromS3Async()` 메서드 추가 - `/analyze-from-s3` 호출
   - `startGenerationAsync()` 메서드 수정 - S3 파일이면 `/generate-from-s3` 호출

### 백엔드 동작 방식
- S3 파일 업로드 시: fileKey를 AI 서버의 `/analyze-from-s3`, `/generate-from-s3`로 전달
- 일반 텍스트 업로드 시: 기존대로 `/analyze`, `/generate`로 전달
- 두 방식 모두 지원 (하위 호환성)

---

## AWS Credentials 공유

AI 서버와 백엔드가 **같은 AWS 계정**의 S3에 접근해야 합니다.

### 백엔드 설정 (application.yml)
```yaml
aws:
  s3:
    bucket: story-game-bucket
    region: ap-northeast-2
  credentials:
    access-key: ${AWS_ACCESS_KEY}
    secret-key: ${AWS_SECRET_KEY}
```

### AI 서버 설정 (.env 또는 환경변수)
```bash
AWS_ACCESS_KEY_ID=same-as-backend
AWS_SECRET_ACCESS_KEY=same-as-backend
AWS_REGION=ap-northeast-2
AWS_S3_BUCKET=story-game-bucket
```

**주의: 같은 credentials를 사용해야 합니다!**

---

## 다음 단계

1. AI 서버 레포지토리로 이동
2. 위 가이드대로 코드 추가
3. 테스트 실행
4. 백엔드와 통합 테스트

백엔드는 준비 완료되었으므로, AI 서버만 수정하면 됩니다!
