"""프로젝트 공통 설정 (경로, 시드, 검증 방식).

팀원 모두 이 값을 그대로 import해서 사용합니다. 값을 바꿔야 하면 PR로 합의 후 변경합니다.
"""
from pathlib import Path

# ===== 경로 =====
# 이 파일(src/config.py) 기준으로 저장소 루트를 찾으므로, 노트북/스크립트 어디서 실행해도 같은 경로가 됩니다.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SUBMISSION_DIR = ROOT_DIR / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# ===== 컬럼 =====
ID_COL = "ID"
TARGET_COL = "SUBCLASS"
WT = "WT"  # 변이 없음(Wild Type) 표기

# ===== 재현성 / 검증 방식 (docs/team_rules.md '검증 방식'과 동일) =====
SEED = 42
N_FOLDS = 5
