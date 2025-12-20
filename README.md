이 프로젝트는 AI를 활용하여 주어진 소설 텍스트를 기반으로 상호작용 가능한 에피소드 형식의 스토리를 생성합니다. 원본 텍스트를 분석하여 등장인물을 추출하고, 게이지(수치) 기반의 게임 플레이 시스템을 설계한 뒤, 여러 에피소드로 이어지는 분기형 스토리 트리를 구축합니다.

주요 기능 (Features)
소설 분석 (Novel Analysis): 이야기 요약, 주요 등장인물 추출, 그리고 이야기의 상태 변화를 추적하기 위한 "게이지" 시스템을 제안합니다.

에피소드 구조 (Episodic Structure): 서사를 뚜렷한 에피소드 단위로 분할합니다.

분기형 서사 (Branching Narratives): LangGraph를 사용하여 각 에피소드마다 선택지가 포함된 스토리 노드 트리(Tree)를 생성합니다.

동적 결말 (Dynamic Endings): 누적된 게이지 값에 따른 다중 최종 엔딩과 플레이어의 선택에 따른 에피소드별 엔딩을 설계합니다.

내보내기 (Exporting): 생성된 스토리를 Markdown, HTML, 또는 게임 엔진에서 사용하기 쉬운 JSON 형식으로 내보냅니다.

설치 및 설정 (Setup)
가상 환경 생성:

Bash

python -m venv venv
Bash

source venv/bin/activate  # Windows의 경우 `venv\Scripts\activate` 사용
필수 라이브러리 설치:

Bash

pip install -r requirements.txt
API 키 설정: 프로젝트 루트 경로에 .env 파일을 생성하고 OpenAI API 키를 추가하세요:

OPENAI_API_KEY="sk-..."
사용법 (Usage)
메인 스크립트를 실행하여 스토리 생성 파이프라인을 시작하세요:

Bash

python main.py
스크립트는 기본적으로 main.py에 포함된 "파리대왕(Lord of the Flies)"의 예시 텍스트를 사용합니다. main.py를 수정하여 파일(예: my_novel.txt)에서 사용자의 소설 텍스트를 불러와 사용하도록 변경할 수 있습니다.

최종 생성된 스토리는 episode_story.json 파일에 저장됩니다.

# test
