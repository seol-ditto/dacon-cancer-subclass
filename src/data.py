"""데이터 로드와 고정 규칙 전처리.

train과 test에 '똑같은 고정 규칙'만 적용합니다. test에서 계산한 통계값은 쓰지 않으므로 Data Leakage가 아닙니다.

1. 빈칸('') → 'WT'  (test에만 237칸, 01 EDA)
2. 변이 표기 통일 (HGVS 표준 표기법 기준, 01 EDA 3)-3)
   - 종결코돈 'X' → '*'     예) R492X  → R492*   ('X'는 예전 표기, 같은 뜻)
   - frameshift 뒤 아미노산 제거 예) K600Nfs → K600fs
3. 이소폼 중복 제거
   - 한 칸 안에서 위치 번호만 다르고 변이 내용(ref/alt)이 같은 토큰은 1개만 남깁니다.
     예) 'R292X R272X R258X' → 'R292*'  (같은 변이를 이소폼별 번호로 여러 번 적은 것)
   - 근거: test TP53은 한 칸 안 위치 차이가 (0, 39, 132, 159)로 수백 명에게서 똑같이 반복됨
     = TP53 이소폼(Δ40, Δ133, Δ160) 번호 차이. 서로 다른 변이였다면 이렇게 일정할 수 없음.
   - 남기는 번호는 가장 큰 값(보통 기본 이소폼 = train 표기).
   - 한계: 같은 유전자에 ref/alt가 같은 '진짜 다른 변이'가 있으면 1개로 합쳐짐
     (train 변이 칸의 0.7%). 유전자 단위 있음/없음·유형 피처에는 영향 없고 '개수'만 조금 줄어듭니다.
"""
import re

import pandas as pd

from src.config import (
    ID_COL,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    WT,
)

_STOP_X = re.compile(r"^([A-Z])(\d+)X$")           # R492X
_FS_AA = re.compile(r"^([A-Z*]*\d+)[A-Z]?fs.*$")   # K600Nfs, K600Nfs*12
_POS = re.compile(r"^([A-Z*]?)(\d+)(.*)$")         # (ref)(위치)(나머지)


def load_data(normalize=True):
    """train, test, sample_submission을 불러와 고정 규칙 전처리를 적용해 반환합니다.

    Parameters
    ----------
    normalize : bool
        True(기본)면 변이 표기 통일 + 이소폼 중복 제거를 train/test 모두에 적용합니다.
        False는 예전 방식과 비교할 때만 사용하세요.

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
    assert list(test.columns) == [ID_COL] + gene_cols, "train/test 유전자 컬럼이 다릅니다."

    for df in (train, test):  # train과 test에 같은 규칙을 같은 순서로 적용
        df[gene_cols] = df[gene_cols].replace("", WT)
        if normalize:
            df[gene_cols] = normalize_mutations(df[gene_cols])

    return train, test, submission


def get_gene_cols(df):
    """ID, SUBCLASS를 제외한 유전자 컬럼 목록 (원본 순서 유지)."""
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]


def normalize_token(token):
    """변이 토큰 1개의 표기를 train 표기로 통일합니다.

    >>> normalize_token('R492X'), normalize_token('K600Nfs'), normalize_token('G12D')
    ('R492*', 'K600fs', 'G12D')
    """
    token = _STOP_X.sub(r"\1\2*", token)
    token = _FS_AA.sub(r"\1fs", token)
    return token


def normalize_cell(cell):
    """한 칸(공백으로 구분된 여러 토큰)을 정규화하고 이소폼 중복을 제거합니다.

    >>> normalize_cell('R292X R272X R258X'), normalize_cell('P348L P348L'), normalize_cell('H286Y D323D')
    ('R292*', 'P348L', 'H286Y D323D')
    >>> normalize_cell('R141H R114H R234H R273H')
    'R273H'
    """
    if cell == WT:
        return cell
    best = {}  # 변이 내용(ref, alt) → 위치 번호가 가장 큰 토큰
    for token in cell.split():
        token = normalize_token(token)
        m = _POS.match(token)
        if not m:
            best.setdefault(token, (0, token))
            continue
        key, pos = (m.group(1), m.group(3)), int(m.group(2))  # 위치 번호를 뺀 '변이 내용'
        if key not in best or pos > best[key][0]:
            # 가장 긴 기본 이소폼(canonical)의 번호가 보통 가장 큼 → train 표기와 맞춤
            # 예) TP53 'R141H R114H R234H R273H' → R273H (train과 같은 번호)
            best[key] = (pos, token)
    return " ".join(t for _, t in best.values())


def normalize_mutations(gene_df):
    """유전자 컬럼 DataFrame 전체에 normalize_cell()을 적용합니다 (WT 칸은 그대로)."""
    return gene_df.apply(lambda col: col.map(normalize_cell))
