# 로드 오브 히어로즈 장비 분석 프로그램

## 사용 전 필요한 사항들

- Python 3.13: 핵심 인터프리터
- uv: 패키지 설치
- Google Play Games Beta를 통해 로드 오브 히어로즈를 받으셔야 합니다.
  * 맥은 아직 안됩니다.
- 듀얼 모니터(Optional): 한쪽 화면에서 조작하면서 콘솔을 보셔야 합니다.

## 어떻게 사용하나요?

0. 기본적인 컴퓨터공학에 대한 지식이 있다는 가정을 하겠습니다.
1. 이 리포지토리를 `git clone`으로 다운로드 받아주세요.
2. 다운로드 받은 소스를 터미널에서 접근한 뒤, `uv sync`를 입력해서 필요한 패키지와 가상환경을 설치해 주세요.
3. `uv run main.py`를 입력하시면 프로그램이 가동된 상태입니다. 이 상태에서 `ctrl + 마우스 좌클릭`을 하시면 캡쳐 함수가 실행되면서 콘솔에 파싱 결과가 나옵니다.
4. `esc`키를 눌러 프로그램을 종료하시면 됩니다.

## QnA

Q1. 배포하실 생각 없나요?
- A1. 기본적으로 없습니다. 소스코드 공개는 하며, 제 코드를 들고와서 다른데서 사용하셔도 딱히 상관은 없습니다만, 본격적으로 누구나 따라할 수 있는 프로그램으로 만들 생각은 없습니다. 기여해주시면 제가 반영해보도록 할게요.

Q2. 이런저런 기능이 필요해요.
- A2. 노력은 해보겠습니다만, 일단은 제가 필요로 해서 만든거인데다가 취미로 만드는 거라서 시간이 걸립니다. Pull Request 환영합니다.

Q3. 이런저런 버그가 있어요.
- A3. Issue에다가 현상 기록해주시면 제가 확인해서 고쳐볼게요. 다만 A2에서도 적었듯이 취미로 만드는 것이라 시간이 걸릴겁니다.
