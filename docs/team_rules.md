# 팀 규칙

## 브랜치 전략

```
main
feature/*
```

- **main**: 최종 코드가 저장되는 브랜치입니다. 직접 push하지 않고 PR로만 병합합니다.
- **feature/**: 개인 작업 브랜치입니다. main에서 브랜치를 만들어 작업한 뒤 commit, 이후 main으로 PR을 보냅니다.

### 브랜치 이름 규칙

```
feature/작업-내용
```

- 영어 소문자와 하이픈(-)만 사용합니다.
- 작업 내용이 한눈에 보이도록 짧게 씁니다.

| 예시 | 작업 |
|---|---|
| `feature/eda` | 탐색적 데이터 분석 |
| `feature/preprocessing` | 데이터 전처리 |
| `feature/lgbm-baseline` | LightGBM 베이스라인 |
| `feature/catboost` | CatBoost 실험 |
| `feature/ensemble` | 앙상블 |

### 작업 순서

1. main 최신화: `git switch main` → `git pull`
2. 브랜치 생성: `git switch -c feature/작업-내용`
3. 작업 후 커밋, push: `git push -u origin feature/작업-내용`
4. 깃허브에서 main으로 PR 생성 → 리뷰 후 병합
5. 병합된 브랜치는 삭제합니다.

## 커밋 메시지 컨벤션

### 형식

```
타입: 작업 내용
```

- 타입은 아래 표의 영어 소문자를 그대로 씁니다.
- 작업 내용은 짧고 명확하게 씁니다.

### 타입

| 타입 | 언제 쓰나요 | 예시 |
|---|---|---|
| feat | 새 기능 추가 (모델, 피처, 전처리 함수 등) | `feat: add mutation one-hot encoding` |
| fix | 버그 수정 | `fix: fix label encoding mismatch` |
| refactor | 동작은 그대로, 코드 구조만 정리 | `refactor: split preprocessing into functions` |
| docs | README 등 문서 수정 | `docs: update README` |
| chore | 설정 파일, 패키지, `.gitignore` 등 | `chore: add .gitignore` |
| exp | 실험 (하이퍼파라미터, 새 모델 시도 등) | `exp: try CatBoost with class weights` |
| perf | 속도 개선 (학습 시간 단축 등) | `perf: reduce features with variance filter` |
| remove | 코드나 파일 삭제 | `remove: delete unused notebook` |
| submit | 리더보드 제출용 버전 | `submit: LGBM baseline (LB 0.xxx)` |

### 해커톤 팁

- `exp`, `submit` 커밋에는 점수(CV, LB)를 같이 적으면 나중에 어떤 버전이 좋았는지 찾기 쉽습니다.
  - 예: `exp: XGBoost depth 6 (CV 0.xxx)`
- 커밋은 작게, **자주** 합니다. **한 커밋에는 한 가지 작업**만 담습니다.

## 데이터 경로

대회 데이터는 git에 올리지 않으므로, **각자 다운로드해서 아래 위치에 그대로 둡니다.**
파일 이름은 바꾸지 않습니다.

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

- 코드에서는 항상 **상대 경로**로 불러옵니다. 개인 컴퓨터의 절대 경로(`/Users/...`, `C:\...`)는 쓰지 않습니다.
  ```python
  DATA_DIR = "data"
  train = pd.read_csv(f"{DATA_DIR}/train.csv")
  ```
- 노트북(`notebooks/`)에서 실행할 때는 `../data/train.csv`로 불러옵니다.

## 데이터 사용 원칙 (**Data Leakage 방지**)

- 인코더, 스케일러 등 전처리는 **train으로만 fit**하고, test에는 transform만 적용합니다.
- test 데이터는 형식 확인과 예측에만 사용합니다. test 분포를 보고 피처를 고르거나 기준을 정하지 않습니다.
- 임계값, 앙상블 구성 등 모든 결정은 검증 데이터(CV) 기준으로 합니다.

## 검증 방식

> **확정 (11/09)**: EDA 결과를 근거로 아래 방식으로 확정합니다.
> - 데이터 크기: train 6,201행 → 5-Fold면 fold당 검증 약 1,240행으로 충분
> - 가장 적은 아형: DLBC 38명 → 5-Fold에서도 fold마다 7~8명씩 검증에 포함됨
> - 아형 불균형: BRCA 786명 vs DLBC 38명 (약 20배) → 아형 비율을 유지하는 Stratified 분할 사용
> - 학습 시간: 로지스틱 회귀 기준 5-Fold 1회 수 초 → 반복 실험에 부담 없음
>
> 공통 함수: `src/validation.py`의 `get_folds()`, `run_cv()`를 사용합니다.

팀원 간 CV 점수를 비교할 수 있도록 **모두 같은 방식으로 검증합니다.**

| 항목 | 값 |
|---|---|
| 분할 방식 | Stratified K-Fold (SUBCLASS 기준) |
| Fold 수 | 5 |
| Shuffle | True |
| Seed | 42 |
| 평가 지표 | Macro F1 score |

```python
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    ...
```

- 모델에도 `random_state=SEED`(또는 `seed`, `random_seed`)를 지정합니다.
- CV 점수는 5개 fold 점수의 평균으로 기록합니다.

## 실험 기록

모든 실험과 제출은 공유 실험 기록표(구글 스프레드시트)에 남깁니다.

- [실험 기록표](https://docs.google.com/spreadsheets/d/1cfUcgF3V7Kw16Vt3tZxV04qgElGCalMTkMXN9T6LZ7k/edit?gid=0#gid=0)

| 날짜 | 담당 | 브랜치 | 모델 | 주요 설정 | CV | LB | 메모 |
|---|---|---|---|---|---|---|---|
| 11/07 | 이름 | feature/lgbm-baseline | LightGBM | 원핫 인코딩, 기본 파라미터 | 0.xxx | 0.xxx | 베이스라인 |

- 리더보드에 제출하지 않은 실험도 CV 점수와 함께 기록합니다.
- 제출 파일은 `submissions/`에 아래 형식으로 저장합니다.
  ```
  날짜_이름_모델_CV점수.csv
  예: 1107_seol_lgbm_0.712.csv
  ```
