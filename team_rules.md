# 팀 규칙

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
