"""
Test script to check if AI server generates intro_text
"""
import requests
import json

# Test data
test_request = {
    "initialAnalysis": {
        "summary": "로미오와 줄리엣의 비극적인 사랑 이야기",
        "characters": [
            {
                "name": "로미오",
                "aliases": ["Romeo"],
                "description": "몬태규 가문의 젊은 귀족",
                "relationships": ["줄리엣의 연인"]
            }
        ]
    },
    "storyConfig": {
        "numEpisodes": 3,
        "maxDepth": 2,
        "selectedGaugeIds": ["love", "conflict"]
    },
    "novelContext": "베로나에서 몬태규와 캐플릿 두 가문이 대립하고 있다. 로미오는 줄리엣을 만나 사랑에 빠진다.",
    "currentEpisodeOrder": 1,
    "previousEpisode": None
}

try:
    print("🔥 Sending test request to AI server...")
    response = requests.post(
        "http://localhost:8000/generate-next-episode",
        json=test_request,
        timeout=300
    )

    if response.status_code == 200:
        data = response.json()
        print("\n✅ Response received!")
        print(f"Episode Title: {data.get('title')}")
        print(f"Intro Text Present: {data.get('intro_text') is not None}")
        print(f"Intro Text Length: {len(data.get('intro_text', ''))}")

        if data.get('intro_text'):
            print(f"\nIntro Text Preview:\n{data['intro_text'][:200]}...")
        else:
            print("\n❌ NO INTRO TEXT IN RESPONSE!")
            print(f"\nResponse keys: {data.keys()}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Request failed: {e}")
