# 🧬 암환자 유전체 변이 기반 암종 분류 AI 모델

> 데이콘 [초격차] AI 헬스케어 7기 해커톤 (2026.11.06 ~ 2026.11.13)

> 📌 작업 전 [팀 규칙](docs/team_rules.md)을 꼭 확인해 주세요!

---

# 프로젝트 개요

암은 유전자 수준에서 발생하는 다양한 변이(mutation)에 의해 아형(subtype)이 결정되며, 아형에 따라 치료 방침과 예후가 크게 달라집니다.
본 프로젝트는 환자별 유전자 변이 프로파일을 활용해 암의 아형(SUBCLASS)을 예측하는 다중 분류(multi-class classification) 모델을 개발하는 것을 목표로 합니다.

- **과제**: 유전자 변이 정보 기반 암 아형(SUBCLASS) 예측
- **문제 유형**: 다중 분류 (Multi-class Classification)
- **평가 지표**: Macro F1 Score

---

# 팀 구성

| 이름 | GitHub |
|---|---|
| 전은지 | 팀장 / [@ellieJ](https://github.com/ellieJ) |
| 강선영 | [@kangkiwi](https://github.com/kangkiwi) |
| 김설아 | [@seol-ditto](https://github.com/seol-ditto) |
| 서민서 | @github ID fill up 예정 |

---

# 데이터 설명

> 대회 규칙에 따라 데이터는 저장소에 포함하지 않습니다.

| 변수 | 설명 |
|---|---|
| ID | 환자 고유 ID |
| SUBCLASS | 암 아형 (예측 대상) |
| 유전자명 (4,384개) | 해당 유전자의 변이 정보 (WT: 변이 없음 / 변이 표기) |

- train 6,201행 / test 2,546행 / 아형 26개
- 전체 값의 99.19%가 WT (매우 희소)
- 변이 표기 예: `G12D`(아미노산 치환), `R895R`(침묵 변이), `R213*`(종결), `K123fs`(프레임시프트). 한 칸에 여러 변이는 공백으로 구분
- test의 빈칸은 `WT`로 채움 (고정 규칙, `src/data.py`)

---

# 데이터 전처리

- (작성 예정)

---

# 모델링

- 사용 모델
  - (작성 예정)

---

# 실험 결과

[실험 기록표](https://docs.google.com/spreadsheets/d/1cfUcgF3V7Kw16Vt3tZxV04qgElGCalMTkMXN9T6LZ7k/edit?gid=0#gid=0)

| 모델 | CV Score | LB Score |
|---|---|---|
| 제공 베이스라인 (OrdinalEncoder + XGBoost) | 0.3051 | - |
| LightGBM (이진화) | 0.2893 | - |
| 로지스틱 회귀 (이진화, C=1) | 0.3432 | 0.2533 |
| **로지스틱 회귀 (이진화, C=0.1)** | **0.3818** | **0.2739** |
| 로지스틱 회귀 (이진화 + 절단 변이 + 변이 개수, C=0.1) | 0.4188 | 0.2706 |

- 검증: Stratified 5-Fold (seed 42), Macro F1 ([팀 규칙](docs/team_rules.md) 참고)
- 상세 기록은 실험 기록표와 `notebooks/` 각 노트북의 결론 셀 참고

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
├── .github/
│   ├── workflows/                 # 디스코드 알림 연동
│   └── pull_request_template.md   # PR 템플릿
├── data/                          # 대회 데이터 (git 제외)
├── docs/
│   ├── meetings/                  # 회의록 (템플릿 포함)
│   └── team_rules.md              # 팀 규칙 (브랜치 전략, 커밋 컨벤션)
├── notebooks/                     # EDA 및 실험 노트북
├── src/                           # 공통 모듈
│   ├── config.py                  # 경로, SEED, Fold 수
│   ├── data.py                    # 데이터 로드, test 빈칸 처리
│   ├── features.py                # 피처 생성 (이진화 등)
│   └── validation.py              # 공통 CV, Macro F1, 제출 파일 저장
├── submissions/                   # 제출 파일 (git 제외)
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
