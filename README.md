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
| 김설아 | 팀장 / [@seol-ditto](https://github.com/seol-ditto) |
| 전은지 | [@ellieJ](https://github.com/ellieJ) |
| 강선영 | [@kangkiwi](https://github.com/kangkiwi) |
| 서민서 | [@smins98](https://github.com/smins98) |

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

### EDA에서 확인한 데이터 특징

1. **라벨이 서로 겹치는 묶음 코호트가 있음**
   - KIPAN = KIRC + KIRP + KICH (신장암 묶음), GBMLGG = GBM + LGG (뇌종양 묶음), STES = 위암 + 식도암 묶음
   - train 안에 변이 프로필이 같은데 라벨만 다른 쌍이 있음 (KIPAN↔KIRC, GBMLGG↔LGG)
   - 이 쌍의 오분류는 모델 문제가 아니라 라벨 구조에서 오는 한계
2. **test는 train과 변이 표기 방식이 다름**
   - 종결 변이: train은 `R213*`, test는 `R492X`
   - 프레임시프트: train은 `M622fs`, test는 `K600Nfs`처럼 아미노산이 하나 더 붙음
   - test는 같은 변이를 이소폼마다 위치 번호만 바꿔 한 칸에 여러 번 적음 (예: `R292X R272X R258X`)
   - 그 결과 환자 1명당 변이 토큰 수가 train 약 41개, test 약 133개로 차이가 남
3. **정규화 후에도 남는 차이**
   - test 환자 1명당 변이 유전자 수가 train의 약 2배 (중앙값 14개 vs 28개)
   - TP53, PIK3CA 같은 대표 암 유전자는 비교적 안정적이지만, 일부 유전자는 test에서 훨씬 자주 나옴
   - test는 암종 비율 자체가 train과 다를 가능성이 있음 (예측 결과에서 COAD 비율이 높게 나옴)

---

# 데이터 전처리

1. **빈칸 처리**: test의 빈칸은 `WT`로 채움
2. **변이 표기 정규화** (`src/data.py`의 `load_data()` 기본값)
   - HGVS 표기 규칙에 맞춰 train과 test에 똑같은 고정 규칙을 적용 (test 통계 사용 안 함)
   - `R492X` → `R492*`, `K600Nfs` → `K600fs`
   - 한 칸 안에서 위치 번호만 다르고 변이 내용(원래 아미노산 → 바뀐 아미노산)이 같은 토큰은 1개로 합침 (이소폼 중복 제거)
   - 적용 후 test 토큰 수 133개 → 80개, 종결 변이 비율도 train과 비슷해짐
   - `dedupe` 옵션: 변이 개수 피처는 `dedupe=True`, 핫스팟 토큰 피처는 대표 번호가 사라지지 않도록 `dedupe=False`로 따로 불러옴
3. **피처 생성** (`src/features.py`)
   - `any`: 유전자별 변이 유무 이진화 (0/1)
   - `stats`: 변이 유형별 개수(log), 변이 유전자 수(log), 침묵 변이 비율
   - `trunc`: 유전자별 절단 변이(종결, 프레임시프트) 유무
   - 핫스팟: 자주 나오는 '유전자_변이' 조합의 0/1 피처 (fold 학습 부분에서만 선택)
4. **Data Leakage 방지 원칙**
   - 피처 선택, 가중치 계산 등 데이터에서 무언가를 배우는 과정은 모두 Pipeline 안에서 fold 학습 부분으로만 수행
   - test의 변이 빈도를 보고 유전자를 빼지 않음
   - 예측 비율을 train 비율에 억지로 맞추는 후처리를 하지 않음
   - 겹치는 라벨(KIPAN↔KIRC 등)을 이용한 예측 교정은 사용하지 않음

---

# 모델링

- 사용 모델
  - 로지스틱 회귀 (LR): `class_weight='balanced'`, C=0.1
  - XGBoost: `BalancedXGB` 래퍼로 fold 학습 부분 기준 balanced sample_weight 적용
    - 최종 설정: max_depth=3, learning_rate=0.1, n_estimators=200, colsample_bytree=0.3
  - LightGBM, RandomForest, ExtraTrees: 모델 섞기 비교용
- 클래스 불균형: 모든 모델에 balanced 가중치 적용
- 모델 섞기: 각 모델의 OOF 확률을 저장해 가중 평균 조합 비교 (14번)

---

# 실험 결과

[실험 기록표](https://docs.google.com/spreadsheets/d/1cfUcgF3V7Kw16Vt3tZxV04qgElGCalMTkMXN9T6LZ7k/edit?gid=0#gid=0)

### 정규화 전

| 모델 | CV Score | LB Score |
|---|---|---|
| 제공 베이스라인 (OrdinalEncoder + XGBoost) | 0.3051 | - |
| LightGBM (이진화) | 0.2893 | - |
| 로지스틱 회귀 (이진화, C=1) | 0.3432 | 0.2533 |
| 로지스틱 회귀 (이진화, C=0.1) | 0.3818 | 0.2739 |
| 로지스틱 회귀 (이진화 + 절단 변이 + 변이 개수, C=0.1) | 0.4188 | 0.2706 |
| XGBoost (any + stats, 기본 설정) | 0.3879 | 0.2995 |
| **XGBoost (any + stats, 튜닝)** | **0.4113** | **0.3206** |
| XGBoost (튜닝 + 핫스팟 min10) | 0.4207 | 0.3166 |
| XGBoost (튜닝 + 희귀 유전자 제거 k20) | 0.4115 | 0.3051 |

### 정규화 후

| 모델 | CV Score | LB Score |
|---|---|---|
| XGBoost (any + stats, 튜닝) | 0.4161 | (11/11 제출 예정) |
| XGBoost (any만, 튜닝) | 0.3579 | - |
| XGBoost (튜닝 + 핫스팟 min10) | 0.4213 | (11/11 제출 예정) |
| 섞기: XGBoost 0.7 + LR 0.3 | 0.4231 (OOF) | (11/11 제출 예정) |
| 섞기: XGBoost + LR + RF 같은 비율 | 0.4246 (OOF) | - |

- 검증: Stratified 5-Fold (seed 42), Macro F1 ([팀 규칙](docs/team_rules.md) 참고)
- 섞기 점수는 fold 평균이 아닌 전체 OOF 기준
- 상세 기록은 실험 기록표와 `notebooks/` 각 노트북의 결론 셀 참고

### 실험에서 얻은 교훈

1. **CV와 LB가 자주 어긋남**: CV는 최고였던 LR(이진화 + 절단 변이 + 변이 개수)이 LB는 가장 낮았음. CV와 LB 차이가 0.09~0.15 수준
2. **피처를 추가해서 오른 CV는 LB로 넘어가지 않고, 과적합을 줄여서 오른 CV는 넘어감**
   - 넘어감: LR → XGBoost 전환(+0.026), 얕은 나무 + colsample 튜닝(+0.021)
   - 안 넘어감: 절단 변이·변이 개수 추가, 핫스팟 추가
3. **원인은 test의 표기 차이였음**: 절단 변이 피처가 LB에서 효과가 없던 이유는 test의 `X` 표기를 절단 변이로 인식하지 못해서였음 → 표기 정규화로 대응
4. **강건성 점검 CV는 LB 순서를 맞히지 못함** (13번): 검증 fold에 가짜 변이를 넣을수록 LB 순서와 반대가 됨 (Spearman −0.05 → −0.70). test와 train의 차이는 단순한 무작위 변이 증가가 아니라고 판단
5. **XGBoost에서 변이 개수 피처(stats)는 매우 중요함**: 빼면 CV −0.058, test 예측이 COAD로 쏠림 (11.6% → 22.1%)
6. **섞기 효과는 대부분 LR에서 나옴**: XGBoost와 예측 불일치율이 가장 높은 모델이 LR(0.422). 같은 부스팅 계열인 LightGBM은 넣으면 오히려 하락

---

# 최종 모델

- (11/11 제출 결과 확인 후 작성 예정)
- 최종 제출 후보 2개
  - 최고 LB 모델 1개
  - CV와 LB가 둘 다 안정적인 모델 1개 (Public LB는 test 일부로만 매겨지므로 최고점 하나에만 걸지 않음)

- 선택 이유
  - (작성 예정)

---

# 실행 방법

- 환경 설치
  ```bash
  pip install -r requirements.txt
  ```

- 데이터 준비
  - 대회 데이터(`train.csv`, `test.csv`, `sample_submission.csv`)를 `data/` 폴더에 넣기

- 모델 학습 및 예측 생성
  - 현재는 노트북 단위로 실행 (`notebooks/` 번호 순서)
  - 각 노트북은 `src/validation.py`의 `run_cv`로 CV를 계산하고, `save_submission`으로 `submissions/`에 제출 파일을 저장
  ```bash
  # 최종 모델 확정 후 재현용 노트북/스크립트 경로 기입 예정
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
│   ├── data.py                    # 데이터 로드, test 빈칸 처리, 변이 표기 정규화
│   ├── features.py                # 피처 생성 (이진화, 변이 유형별 개수 등)
│   └── validation.py              # 공통 CV, Macro F1, 제출 파일 저장
├── submissions/                   # 제출 파일 (git 제외)
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 사용 기술

- Language: Python
- Library: pandas, NumPy, scikit-learn, XGBoost, LightGBM, SciPy, Matplotlib, Seaborn
- Collaboration: GitHub, Discord

---

# 향후 개선 방향

1. **프레임시프트 위치 차이 보정**: 정규화 후에도 train과 test의 `fs` 위치가 1~3칸 어긋나는 경우가 있음 → 우리 데이터로 집계해 근거를 확인한 뒤 위치 허용 범위를 둔 매칭 검토
2. **시드 평균**: 가장 좋은 조합에 여러 시드 결과를 평균해 예측 안정화
3. **노이즈에 강한 학습**: fold 학습 부분에서만 train 변이 빈도 기반 노이즈를 추가하는 증강 검토
4. **라벨 구조 반영**: 묶음 코호트(KIPAN, GBMLGG, STES) 구조를 고려한 계층형 분류 검토
5. **최종 재현 코드 정리**: 노트북 실험을 학습·예측 스크립트로 정리
