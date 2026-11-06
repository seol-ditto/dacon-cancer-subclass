# 개발환경 세팅 가이드

> 팀원 모두 **Python 3.12 + 저장소의 `requirements.txt`** 로 환경을 통일합니다.
> 버전이 다르면 경고·오류가 나거나 결과가 재현되지 않을 수 있습니다 (대회 코드 검증 항목: 재현성, 라이브러리 버전 기재).

---

## 1. 저장소 최신 내용 받기

처음이라면 clone합니다.

```bash
git clone <저장소URL> dacon-cancer-subclass
```

이미 받아 두었다면 pull합니다.

```bash
git pull origin main
```

## 2. 저장소 폴더로 이동

```bash
cd dacon-cancer-subclass
```

## 3. Python 3.12로 가상환경 만들기

macOS는 아래 명령입니다.

```bash
python3.12 -m venv .venv
```

Windows는 아래 명령입니다.

```bash
py -3.12 -m venv .venv
```

> `python3.12`가 없다는 오류가 나면 Python 3.12부터 설치하세요. 맥은 `brew install python@3.12`, Windows는 python.org에서 설치하면 됩니다.

## 4. 가상환경 켜기

macOS는 아래 명령입니다.

```bash
source .venv/bin/activate
```

Windows PowerShell은 아래 명령입니다.

```bash
.venv\Scripts\Activate.ps1
```

프롬프트 앞에 `(.venv)`가 보이면 켜진 것입니다.

> 가상환경은 **터미널 창마다 따로** 켜야 합니다. 터미널을 새로 열면 4번만 다시 실행하세요. 끄려면 `deactivate`를 입력합니다.

## 5. 패키지 일괄 설치

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

## 6. macOS만: OpenMP 설치 (XGBoost와 LightGBM에 필요)

```bash
brew install libomp
```

> 설치하지 않으면 `import xgboost`에서 `XGBoostError: libxgboost.dylib could not be loaded` 오류가 납니다.

## 7. 설치 확인

```bash
python -c "import pandas, sklearn, xgboost, lightgbm; print(pandas.__version__, sklearn.__version__, xgboost.__version__, lightgbm.__version__)"
```

`3.1.1 1.9.1 3.4.1 4.7.0`이 나오면 성공입니다.

## 8. 노트북 커널 선택

VS Code나 Jupyter 오른쪽 위의 커널 선택에서 **`.venv (Python 3.12)`** 를 고르세요.

---

## 디스코드 공지용 문구 (복붙)

```
📢 개발환경 통일 안내
1) git pull origin main
2) python3.12 -m venv .venv   (Windows: py -3.12 -m venv .venv)
3) source .venv/bin/activate  (Windows: .venv\Scripts\Activate.ps1)
4) pip install -r requirements.txt
5) 맥 사용자는 brew install libomp 필수!
6) 노트북 커널을 .venv (Python 3.12)로 선택
✅ 확인: python -c "import xgboost, lightgbm; print('OK')"
⚠️ 새 패키지가 필요하면 직접 설치하지 말고 팀장에게 말해 주세요 (requirements.txt 갱신 후 공지)
```

---

## 운영 규칙

- 패키지를 추가할 때는 한 사람이 `requirements.txt`에 **버전을 고정해서** 추가하고 PR로 올립니다. 다른 팀원들은 pull한 뒤 `pip install -r requirements.txt`만 다시 실행하면 됩니다.
- `pip freeze > requirements.txt`로 덮어쓰지 않습니다. 의존 패키지까지 수십 줄이 들어가므로 **직접 쓰는 패키지만** 적습니다.
- `.venv/`는 `.gitignore`에 포함되어 있으니 커밋하지 않습니다.

## 자주 나는 문제

| 증상 | 원인 | 해결 |
| :--- | :--- | :--- |
| `pip install sklearn` 실패 | 패키지 이름 오류 | `scikit-learn`으로 설치 (`requirements.txt`에는 이미 올바르게 적혀 있음) |
| `XGBoostError ... libomp.dylib` | 맥에 OpenMP 없음 | `brew install libomp` 후 커널 재시작 |
| `ModuleNotFoundError` | 노트북 커널이 다른 환경 | 커널을 `.venv (Python 3.12)`로 변경 |
| `(.venv)` 표시가 사라짐 | 새 터미널을 열었음 | `source .venv/bin/activate` 다시 실행 |
