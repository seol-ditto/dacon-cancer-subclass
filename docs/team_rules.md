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
이모지 타입: 작업 내용
```

- 이모지와 타입을 함께 씁니다.
- 작업 내용은 짧고 명확하게 씁니다.

### 타입

| 이모지 | 타입 | 언제 쓰나요 | 예시 |
|:---:|---|---|---|
| ✨ | feat | 새 기능 추가 (모델, 피처, 전처리 함수 등) | `✨ feat: add mutation one-hot encoding` |
| 🐛 | fix | 버그 수정 | `🐛 fix: fix label encoding mismatch` |
| ♻️ | refactor | 동작은 그대로, 코드 구조만 정리 | `♻️ refactor: split preprocessing into functions` |
| 📝 | docs | README 등 문서 수정 | `📝 docs: update README` |
| 🔧 | chore | 설정 파일, 패키지, `.gitignore` 등 | `🔧 chore: add .gitignore` |
| ⚗️ | exp | 실험 (하이퍼파라미터, 새 모델 시도 등) | `⚗️ exp: try CatBoost with class weights` |
| ⚡️ | perf | 속도 개선 (학습 시간 단축 등) | `⚡️ perf: reduce features with variance filter` |
| 🔥 | remove | 코드나 파일 삭제 | `🔥 remove: delete unused notebook` |
| 🚀 | submit | 리더보드 제출용 버전 | `🚀 submit: LGBM baseline (LB 0.xxx)` |

### 해커톤 팁

- `exp`, `submit` 커밋에는 점수(CV, LB)를 같이 적으면 나중에 어떤 버전이 좋았는지 찾기 쉽습니다.
  - 예: `⚗️ exp: XGBoost depth 6 (CV 0.xxx)`
- 커밋은 작게, 자주 합니다. 한 커밋에는 한 가지 작업만 담습니다.
