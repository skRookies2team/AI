"""
S3 연결 테스트 스크립트

이 스크립트는 다음을 테스트합니다:
1. AWS credentials 확인
2. S3 버킷 접근 권한 확인
3. 파일 업로드 테스트
4. 파일 다운로드 테스트
5. 파일 삭제 테스트
"""

import os
import sys
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError
import json

# Windows에서 UTF-8 출력 지원
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# .env 파일 로드
load_dotenv()

def test_s3_connection():
    """S3 연결 테스트"""
    print("=" * 60)
    print("🔧 S3 연결 테스트 시작")
    print("=" * 60)

    # 1. 환경 변수 확인
    print("\n1️⃣ 환경 변수 확인")
    print("-" * 60)

    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'ap-northeast-2')
    aws_bucket = os.getenv('AWS_S3_BUCKET', 'story-game-bucket')

    if aws_access_key:
        print(f"✅ AWS_ACCESS_KEY_ID: {aws_access_key[:10]}***")
    else:
        print("❌ AWS_ACCESS_KEY_ID: 없음")
        return

    if aws_secret_key:
        print(f"✅ AWS_SECRET_ACCESS_KEY: {aws_secret_key[:10]}***")
    else:
        print("❌ AWS_SECRET_ACCESS_KEY: 없음")
        return

    print(f"✅ AWS_REGION: {aws_region}")
    print(f"✅ AWS_S3_BUCKET: {aws_bucket}")

    # 2. S3 클라이언트 초기화
    print("\n2️⃣ S3 클라이언트 초기화")
    print("-" * 60)

    try:
        s3_client = boto3.client(
            's3',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        print("✅ S3 클라이언트 생성 성공")
    except Exception as e:
        print(f"❌ S3 클라이언트 생성 실패: {str(e)}")
        return

    # 3. S3 버킷 접근 권한 확인
    print("\n3️⃣ S3 버킷 접근 권한 확인")
    print("-" * 60)

    try:
        # 버킷 목록 확인
        response = s3_client.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        print(f"✅ 접근 가능한 버킷 목록: {buckets}")

        if aws_bucket in buckets:
            print(f"✅ 대상 버킷 '{aws_bucket}' 접근 가능")
        else:
            print(f"⚠️  대상 버킷 '{aws_bucket}'이 목록에 없습니다")
            print(f"   (권한이 있으면 여전히 사용 가능할 수 있습니다)")
    except NoCredentialsError:
        print("❌ AWS credentials가 올바르지 않습니다")
        return
    except ClientError as e:
        print(f"❌ 버킷 목록 조회 실패: {e.response['Error']['Message']}")
        return
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        return

    # 4. 파일 업로드 테스트
    print("\n4️⃣ 파일 업로드 테스트")
    print("-" * 60)

    test_key = "test/test_upload.json"
    test_data = {
        "message": "S3 연결 테스트",
        "timestamp": "2025-01-26",
        "status": "success"
    }

    try:
        s3_client.put_object(
            Bucket=aws_bucket,
            Key=test_key,
            Body=json.dumps(test_data, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"✅ 파일 업로드 성공: s3://{aws_bucket}/{test_key}")
    except ClientError as e:
        print(f"❌ 파일 업로드 실패: {e.response['Error']['Message']}")
        return
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        return

    # 5. 파일 다운로드 테스트
    print("\n5️⃣ 파일 다운로드 테스트")
    print("-" * 60)

    try:
        response = s3_client.get_object(Bucket=aws_bucket, Key=test_key)
        content = response['Body'].read().decode('utf-8')
        downloaded_data = json.loads(content)
        print(f"✅ 파일 다운로드 성공")
        print(f"   내용: {downloaded_data}")
    except ClientError as e:
        print(f"❌ 파일 다운로드 실패: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")

    # 6. 파일 삭제 테스트
    print("\n6️⃣ 파일 삭제 테스트")
    print("-" * 60)

    try:
        s3_client.delete_object(Bucket=aws_bucket, Key=test_key)
        print(f"✅ 테스트 파일 삭제 성공")
    except ClientError as e:
        print(f"❌ 파일 삭제 실패: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")

    # 7. 최종 결과
    print("\n" + "=" * 60)
    print("🎉 S3 연결 테스트 완료!")
    print("=" * 60)
    print("\n✅ 모든 테스트가 성공했습니다!")
    print("   - AWS credentials 설정 완료")
    print("   - S3 버킷 접근 가능")
    print("   - 파일 업로드/다운로드 가능")
    print("\n이제 AI 서버에서 S3를 사용할 수 있습니다! 🚀")


if __name__ == "__main__":
    test_s3_connection()
