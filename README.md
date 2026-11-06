# 🧬 암환자 유전체 변이 기반 암종 분류 AI 모델

> 데이콘 [초격차] AI 헬스케어 7기 해커톤
> 기간: 2026.11.06 ~ 2026.11.13

---

# 프로젝트 개요

암은 유전자 수준에서 발생하는 다양한 변이(mutation)에 의해 아형(subtype)이 결정되며, 아형에 따라 치료 방침과 예후가 크게 달라집니다.
본 프로젝트는 환자별 유전자 변이 프로파일을 활용해 암의 아형(SUBCLASS)을 예측하는 다중 분류(multi-class classification) 모델을 개발하는 것을 목표로 합니다.

- **과제**: 유전자 변이 정보 기반 암 아형(SUBCLASS) 예측
- **문제 유형**: 다중 분류 (Multi-class Classification)
- **평가 지표**: (대회 규칙 확인 후 작성)

---

# 팀 구성

| 이름 | 역할 |
|---|---|
| 팀원1 | 팀장 / (역할) |
| 팀원2 | (역할) |
| 팀원3 | (역할) |
| 팀원4 | (역할) |

---

# 데이터 설명

> 대회 규칙에 따라 데이터는 저장소에 포함하지 않습니다.

| 변수 | 설명 |
|---|---|
| ID | 환자 고유 ID |
| SUBCLASS | 암 아형 (예측 대상) |
| 유전자명 (6,199개) | 해당 유전자의 변이 정보 (WT: 변이 없음 / 변이 표기) |

---

# 데이터 전처리

- (작성 예정)

---

# 모델링

- 사용 모델
  - (작성 예정)

---

# 실험 결과

| 모델 | CV Score | LB Score |
|---|---|---|
| | | |

---

# 최종 모델

- (작성 예정)

- 선택 이유
  - (작성 예정)

---

# 실행 방법

- 환경 설치
  ```bash
  pip install -r requirements.txt
  ```

- 모델 학습
  ```bash
  # (작성 예정)
  ```

- 예측 생성
  ```bash
  # (작성 예정)
  ```

---

# 프로젝트 구조

```
dacon-cancer-subclass/
├── .github/workflows/   # 디스코드 알림 연동
├── data/                # 대회 데이터 (git 제외)
├── docs/
│   └── meetings/        # 회의록 (템플릿 포함)
│   └── team_rules.md        # 팀 규칙 (커밋 컨벤션)
├── notebooks/           # EDA 및 실험 노트북
├── src/                 # 전처리 · 학습 · 예측 코드
├── submissions/         # 제출 파일 (git 제외)
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 사용 기술

- Language: Python
- Library: (작성 예정)
- Collaboration: GitHub, Discord

---

# 향후 개선 방향

- (작성 예정)
