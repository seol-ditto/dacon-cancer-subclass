"""데이터 로드와 고정 규칙 전처리.

- test의 빈칸('')은 EDA에서 확인한 대로 'WT'로 채웁니다.
  test 통계값을 쓰지 않는 고정 규칙이므로 Data Leakage가 아닙니다.
"""
import pandas as pd

from src.config import (
    ID_COL,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    WT,
)


def load_data():
    """train, test, sample_submission을 불러오고 빈칸을 'WT'로 채워 반환합니다.

    Returns
    -------
    train, test, submission : pd.DataFrame
    """
    # keep_default_na=False: 'NA' 같은 문자열이 결측으로 바뀌는 것을 막고, 빈칸은 ''로 읽습니다.
    read = dict(dtype=str, keep_default_na=False)
    train = pd.read_csv(TRAIN_PATH, **read)
    test = pd.read_csv(TEST_PATH, **read)
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    gene_cols = get_gene_cols(train)
    train[gene_cols] = train[gene_cols].replace("", WT)
    test[gene_cols] = test[gene_cols].replace("", WT)

    assert list(test.columns) == [ID_COL] + gene_cols, "train/test 유전자 컬럼이 다릅니다."
    return train, test, submission


def get_gene_cols(df):
    """ID, SUBCLASS를 제외한 유전자 컬럼 목록 (원본 순서 유지)."""
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
