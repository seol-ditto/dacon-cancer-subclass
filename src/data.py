"""데이터 로드와 전처리.

train과 test에 '똑같은 규칙'을 적용합니다. test에서 계산한 값과 test 라벨은 쓰지 않으므로 Data Leakage가 아닙니다.
(이소폼 번호 선택에 쓰는 '토큰 사전'은 train으로만 만듭니다.)

1. 빈칸('') → 'WT'  (test에만 237칸, 01 EDA 2))
2. 변이 표기 통일 (HGVS 표준 표기법 기준, 01 EDA 3)-3, 3)-4)
   - 정지코돈 'X' → '*' (토큰 안 모든 X)  예) R492X → R492*, X541X → *541*, X127C → *127C
     'X'는 예전 표기로 '*'와 같은 뜻. train에는 X가 한 글자도 없어 train 결과는 그대로입니다.
   - frameshift 뒤 아미노산 제거          예) K600Nfs → K600fs
3. 이소폼 중복 제거 (01 EDA 3)-5)
   - 한 칸 안에서 위치 번호만 다르고 변이 내용(ref/alt)이 같은 토큰은 1개만 남깁니다.
     예) TP53 'R141H R114H R234H R273H' → 'R273H'  (같은 변이를 이소폼별 번호로 여러 번 적은 것)
   - 근거: test에서 한 칸 안 위치 차이 패턴이 유전자마다 수백 명에게 똑같이 반복됩니다.
     서로 다른 변이였다면 간격이 이렇게 일정할 수 없습니다.
       APC (0, 8, 18, 36, …, 301) 599칸 / TP53 (0, 39, 132, 159) 522칸 / CACNA1A (0, 1) 315칸
       PLEC (0, 110, 133, …) 198칸 / RUNX1 (0, 27) 180칸
     TP53 패턴은 TP53 이소폼(Δ40, Δ133, Δ160)의 번호 차이와 정확히 일치합니다.
   - 남기는 번호: ① train에 같은 유전자·같은 토큰이 있으면 그 번호 ② 없으면 가장 큰 번호
     '가장 큰 번호 = 기본 이소폼'만 쓰면 APC(315칸), BRAF(57칸)에서 train 표기와 어긋나서
     train 사전을 우선합니다. 예) BRAF V600E가 V640E 등으로 남지 않음 → 토큰 단위 핫스팟 피처와 맞음
   - 한계 (모두 피처 영향 작음)
     · train에서도 ref/alt가 같은 '진짜 다른 변이'가 1개로 합쳐짐: 1,593칸(836개 유전자, 변이 칸의 0.7%),
       패턴 없이 흩어져 있음. 유전자 단위 있음/없음·유형 피처는 그대로이고 '개수'만 조금 줄어듭니다.
     · fs는 키가 (ref, 'fs')라 같은 아미노산의 서로 다른 fs(K600fs, K120fs)도 합쳐짐 (train 10여 칸)
     · E746_A750del처럼 번호가 두 개인 토큰은 두 번째 번호가 키에 남아, 이소폼이어도 합쳐지지 않음
"""
import re
from collections import defaultdict

import pandas as pd

from src.config import (
    ID_COL,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    WT,
)

_FS_AA = re.compile(r"^([A-Z*]*\d+)[A-Z]?fs.*$")   # K600Nfs, K600Nfs*12
_POS = re.compile(r"^([A-Z*]?)(\d+)(.*)$")         # (ref)(위치)(나머지)


def load_data(normalize=True, dedupe=True):
    """train, test, sample_submission을 불러와 전처리를 적용해 반환합니다.

    Parameters
    ----------
    normalize : bool
        True(기본)면 변이 표기 통일(X → *, fs 뒤 아미노산 제거)을 train/test 모두에 적용합니다.
        False는 예전 방식과 비교할 때만 사용하세요.
    dedupe : bool
        True(기본)면 이소폼 중복 제거도 적용합니다 (normalize=True일 때만 의미 있음).
        - stats(변이 '개수')를 쓰는 모델 → True 권장: test 개수가 이소폼 중복으로 부풀려진 것을 바로잡음
        - 토큰 단위 핫스팟(11)을 쓰는 모델 → False 권장
          · 환자별 토큰을 set으로 모으면 기본 이소폼 번호(R273H 등)가 이미 들어 있어 중복 제거가 필요 없음
          · train에서는 '진짜 다른 변이'가 합쳐져 핫스팟 후보가 사라짐
            (10명 이상 후보 120 → 116개: LRIG1_L24V, PLEC_A2106A, APC_R876*, PIK3CA_E453K)

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

    for df in (train, test):
        df[gene_cols] = df[gene_cols].replace("", WT)

    if normalize:
        vocab = build_token_vocab(train[gene_cols]) if dedupe else None  # train으로만 만듦
        for df in (train, test):  # train과 test에 같은 규칙을 같은 순서로 적용
            df[gene_cols] = normalize_mutations(df[gene_cols], vocab, dedupe=dedupe)

    return train, test, submission


def get_gene_cols(df):
    """ID, SUBCLASS를 제외한 유전자 컬럼 목록 (원본 순서 유지)."""
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]


def normalize_token(token):
    """변이 토큰 1개의 표기를 train 표기로 통일합니다.

    >>> [normalize_token(t) for t in ['R492X', 'X541X', 'X127C', 'K600Nfs', 'G12D']]
    ['R492*', '*541*', '*127C', 'K600fs', 'G12D']
    """
    token = token.replace("X", "*")   # 앞·뒤·delins 안의 X 모두 정지코돈
    token = _FS_AA.sub(r"\1fs", token)
    return token


def build_token_vocab(gene_df):
    """유전자별로 등장한 (정규화된) 토큰 집합. 반드시 train으로만 만듭니다.

    Returns
    -------
    dict : {유전자명: set(토큰)}
    """
    vocab = defaultdict(set)
    for gene in gene_df.columns:
        col = gene_df[gene]
        for cell in col[col != WT]:
            vocab[gene].update(normalize_token(t) for t in cell.split())
    return dict(vocab)


def normalize_cell(cell, known=frozenset()):
    """한 칸(공백으로 구분된 여러 토큰)을 정규화하고 이소폼 중복을 제거합니다.

    Parameters
    ----------
    cell : str
    known : set
        같은 유전자에서 train에 등장한 토큰 집합. 이소폼 중 이 집합에 있는 번호를 우선 남깁니다.

    >>> normalize_cell('R141H R114H R234H R273H'), normalize_cell('P348L P348L'), normalize_cell('H286Y D323D')
    ('R273H', 'P348L', 'H286Y D323D')
    >>> normalize_cell('V640E V600E V603E', known={'V600E'}), normalize_cell('V640E V600E V603E')
    ('V600E', 'V640E')
    """
    if not isinstance(cell, str) or cell == WT:  # NaN 등 다른 경로로 읽은 데이터 방어
        return cell
    best = {}  # 변이 내용(ref, alt) → (train에 있음, 위치 번호, 토큰)
    for token in cell.split():
        token = normalize_token(token)
        m = _POS.match(token)
        if not m:
            best.setdefault(token, (False, 0, token))
            continue
        key = (m.group(1), m.group(3))  # 위치 번호를 뺀 '변이 내용'
        cand = (token in known, int(m.group(2)), token)
        if key not in best or cand[:2] > best[key][:2]:  # ① train에 있는 번호 ② 큰 번호 순으로 우선
            best[key] = cand
    return " ".join(t for *_, t in best.values())


def normalize_tokens_only(cell):
    """중복 제거 없이 토큰 표기만 통일합니다 (dedupe=False용).

    >>> normalize_tokens_only('R141H R114H R234H R273H'), normalize_tokens_only('R292X R272X')
    ('R141H R114H R234H R273H', 'R292* R272*')
    """
    if not isinstance(cell, str) or cell == WT:
        return cell
    return " ".join(normalize_token(t) for t in cell.split())


def normalize_mutations(gene_df, vocab=None, dedupe=True):
    """유전자 컬럼 DataFrame 전체에 정규화를 적용합니다 (WT 칸은 그대로).

    vocab : build_token_vocab(train) 결과. None이면 '가장 큰 번호'만 사용합니다.
    dedupe : False면 이소폼 중복 제거 없이 표기 통일만 합니다.
    """
    if not dedupe:
        return gene_df.apply(lambda col: col.map(normalize_tokens_only))
    vocab = vocab or {}
    return gene_df.apply(lambda col: col.map(lambda c: normalize_cell(c, vocab.get(col.name, frozenset()))))
