"""피처 생성 함수.

여기 있는 함수는 모두 '한 행(환자) 안에서만' 계산하므로 test에 적용해도 Data Leakage가 아닙니다.
여러 샘플의 통계(빈도, 평균 등)가 필요한 피처는 fit/transform 클래스로 따로 만들고,
반드시 train(또는 fold의 학습 부분)에서만 fit합니다.
"""
import numpy as np
import pandas as pd

from src.config import WT


def to_binary(df, gene_cols):
    """L0 이진화: WT → 0, 그 외(변이 있음) → 1.

    Returns
    -------
    pd.DataFrame (int8), 컬럼 = gene_cols
    """
    return (df[gene_cols] != WT).astype(np.int8)


def add_row_stats(X_bin):
    """행 단위 요약 피처: 총 변이 유전자 수(log1p).

    X_bin : to_binary() 결과
    """
    out = pd.DataFrame(index=X_bin.index)
    out["n_mut_log"] = np.log1p(X_bin.sum(axis=1)).astype(np.float32)
    return out
