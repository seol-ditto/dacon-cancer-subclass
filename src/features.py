"""피처 생성 함수.

여기 있는 함수는 모두 '한 행(환자) 안에서만' 계산하므로 test에 적용해도 Data Leakage가 아닙니다.
여러 샘플의 통계(빈도, 평균 등)가 필요한 피처는 fit/transform 클래스로 따로 만들고,
반드시 train(또는 fold의 학습 부분)에서만 fit합니다.
"""
import re

import numpy as np
import pandas as pd

from src.config import WT

# 변이 유형 (mutation_type()이 돌려주는 값)
MUT_TYPES = ["silent", "missense", "nonsense", "frameshift", "inframe_indel", "stop_loss", "other"]
TRUNC_TYPES = {"nonsense", "frameshift"}  # 단백질이 잘리는(기능을 잃는) 변이

_SNV = re.compile(r"([A-Z*])(\d+)([A-Z*])")  # G12D, R895R, R213*, *261*, (R492X)

# 아미노산 성질 그룹 (11/11 회의: 프라탁신 논문 기준 6그룹 + 종결코돈, 도메인 지식이므로 외부 데이터 아님)
#   근거: Puglisi R. Protein Mutations and Stability, a Link with Disease: The Case Study of Frataxin.
#         Biomedicines 10(2):425 (2022) — 소수성↔친수성 교체, 전하 변화, 글리신·프롤린 도입을 각각 구조 영향으로 설명
#   - 시스테인(C)은 논문 설명("a cysteine that is hydrophobic")대로 소수성
#   - 티로신(Y)은 수산기(-OH)가 있어 친수성 (일반적인 교과서 분류)
AA_GROUPS = {
    "hydrophobic": "AVLIMFWC",   # 물을 싫어하는 것
    "hydrophilic": "STNQY",      # 물을 좋아하는 것 (전하 없음)
    "positive": "KRH",           # 플러스 전기
    "negative": "DE",            # 마이너스 전기
    "glycine": "G",              # 글리신 (가장 작고 유연해서 구조에 특수)
    "proline": "P",              # 프롤린 (고리 구조라 단단해서 구조를 끊음)
    "stop": "*",                 # 종결코돈
}
AA_GROUP = {aa: g for g, members in AA_GROUPS.items() for aa in members}


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


def mutation_type(token):
    """변이 토큰 1개를 유형으로 분류합니다. (01 EDA의 mutation_type()과 같은 규칙)

    예) 'R895R' → silent, 'G12D' → missense, 'R213*' → nonsense, 'K123fs' → frameshift,
        '1529_1530AL>AL' → silent (연속 아미노산 표기), '*261*' → silent (정지코돈 유지)

    Returns
    -------
    str : MUT_TYPES 중 하나
    """
    if "fs" in token:
        return "frameshift"
    if "del" in token or "ins" in token:
        return "inframe_indel"
    if ">" in token:  # 연속 아미노산 치환 (예: 1529_1530AL>AL)
        before, after = token.split(">", 1)
        ref = before.lstrip("0123456789_")
        if ref == after:
            return "silent"
        if "*" in after and "*" not in ref:
            return "nonsense"
        return "missense"
    m = _SNV.fullmatch(token)
    if m:
        ref, alt = m.group(1), m.group(3)
        if ref == alt:
            return "silent"
        if alt in ("*", "X"):  # 'X'도 종결코돈 (보통은 load_data()에서 '*'로 바뀌어 들어옴)
            return "nonsense"
        if ref in ("*", "X"):  # 앞의 X도 정지코돈 (예: X127C)
            return "stop_loss"
        return "missense"
    return "other"


def aa_substitution(token):
    """단일 아미노산 치환 토큰 → '원래>바뀐' 아미노산 문자열. 한 행 안에서만 계산 → 누수 아님.

    침묵·넌센스·stop_loss도 같은 형식으로 돌려줌 (필요하면 mutation_type()으로 걸러서 사용).
    frameshift, in-frame indel, 여러 아미노산 치환(1529_1530AL>AL)은 None.

    >>> [aa_substitution(t) for t in ['R132H', 'G12D', 'R895R', 'R213*', '*261*', 'K600fs', 'P233del', '1529_1530AL>AL']]
    ['R>H', 'G>D', 'R>R', 'R>*', '*>*', None, None, None]
    """
    m = _SNV.fullmatch(token.replace("X", "*"))   # 정규화 전 데이터의 X 표기도 처리
    return f"{m.group(1)}>{m.group(3)}" if m else None


def aa_group_change(token):
    """단일 아미노산 치환 토큰 → '원래 그룹>바뀐 그룹' (AA_GROUPS 기준). 치환이 아니면 None.

    >>> [aa_group_change(t) for t in ['R132H', 'G12D', 'E545K', 'V600E', 'L156P', 'R213*', 'K600fs']]
    ['positive>positive', 'glycine>negative', 'negative>positive', 'hydrophobic>negative', 'hydrophobic>proline', 'positive>stop', None]
    """
    sub = aa_substitution(token)
    if sub is None:
        return None
    ref, alt = sub.split(">")
    if ref not in AA_GROUP or alt not in AA_GROUP:   # 표준 20개 + 종결코돈 외 문자 (안전장치)
        return None
    return f"{AA_GROUP[ref]}>{AA_GROUP[alt]}"


def mutation_features(df, gene_cols):
    """변이 유형 기반 피처를 한 번에 만듭니다. 한 칸에 여러 변이는 공백으로 구분되어 있습니다.

    Returns
    -------
    dict of pd.DataFrame (index = df.index)
        any       : 변이 있음 (= to_binary), 컬럼 = 유전자명
        nonsilent : 침묵이 아닌 변이가 1개 이상, 컬럼 = 유전자명
        silent    : 침묵 변이가 1개 이상, 컬럼 = '유전자__silent'
        trunc     : 절단 변이(nonsense, frameshift)가 1개 이상, 컬럼 = '유전자__trunc'
        stats     : 행 단위 요약 (유형별 변이 수 log1p, 변이 유전자 수 log1p, 침묵 변이 비율)
        ratio     : 행 단위 요약 비율판 (22a) — 유형별 변이 비율(유형 수 / 전체 변이 수) + 전체 변이 수 log1p
                    · in-frame indel은 분모·열에서 제외 (train 3개 / test 3,501개, 표기 차이 아님 → 22 노트북 확인)
                    · 변이가 2배여도 같은 암이면 비율은 비슷 → test 변이 수 증가에 덜 흔들리게
                    · 개수는 전체 변이 수 하나만 log로 남김, 변이 0개 환자는 비율 모두 0

    침묵 변이 처리 3가지
        ① 포함 : any                 ← 팀 기본값 (08: ① > ③ 분리 > ② 제외)
        ② 제외 : nonsilent
        ③ 분리 : nonsilent + silent

    trunc는 CV를 올리지만(+0.008) LB 효과는 없었습니다 (07 F).
        → test는 종결코돈을 'X'로 표기해 nonsense가 trunc에서 빠져 있었음 (test trunc 1.37‰ → 수정 후 1.88‰).
          load_data(normalize=True)로 고친 뒤 LB에서 다시 확인하세요.
    stats(변이 '개수' 피처)는 XGB에서 중요한 피처입니다 (15).
        → XGB에서 빼면 CV −0.058 (0.4161 → 0.3579), test 예측이 COAD로 쏠림 (22.1%, train 3.6%)
        → 10 + 정규화(any + stats)는 LB 0.3202로 정규화 전 최고(0.3206)와 비슷 → 지금 기준 모델은 stats 포함
        단, test는 정규화 후에도 환자당 변이 수가 train의 약 2배라서(중앙값 14 vs 28) '개수'는 분포가 어긋남
        → 개수 대신 비율로 바꾸는 실험 진행 중 (22)
    """
    V = df[gene_cols].to_numpy()
    n, m = V.shape
    nonsilent = np.zeros((n, m), np.int8)
    silent = np.zeros((n, m), np.int8)
    trunc = np.zeros((n, m), np.int8)
    counts = np.zeros((n, len(MUT_TYPES)), np.float32)
    type_idx = {t: k for k, t in enumerate(MUT_TYPES)}

    rows, cols = np.nonzero(V != WT)
    for i, j in zip(rows, cols):
        for token in V[i, j].split():
            t = mutation_type(token)
            counts[i, type_idx[t]] += 1
            if t == "silent":
                silent[i, j] = 1
            else:
                nonsilent[i, j] = 1
            if t in TRUNC_TYPES:
                trunc[i, j] = 1

    idx = df.index
    X_any = to_binary(df, gene_cols)
    stats = pd.DataFrame(np.log1p(counts), index=idx, columns=[f"n_{t}_log" for t in MUT_TYPES])
    stats["n_gene_log"] = np.log1p(X_any.sum(axis=1)).astype(np.float32)
    stats["silent_ratio"] = counts[:, type_idx["silent"]] / np.maximum(counts.sum(axis=1), 1)

    # in-frame indel은 train에 거의 없고(3개) test에만 많음(3,501개) → 분모·열에서 제외 (22 노트북 확인 셀)
    keep = [k for k, t in enumerate(MUT_TYPES) if t != "inframe_indel"]
    total = counts[:, keep].sum(axis=1, keepdims=True)
    ratio = pd.DataFrame(counts[:, keep] / np.maximum(total, 1), index=idx,
                         columns=[f"r_{MUT_TYPES[k]}" for k in keep])
    ratio["n_mut_log"] = np.log1p(total[:, 0])

    return {
        "any": X_any,
        "nonsilent": pd.DataFrame(nonsilent, index=idx, columns=gene_cols),
        "silent": pd.DataFrame(silent, index=idx, columns=[f"{g}__silent" for g in gene_cols]),
        "trunc": pd.DataFrame(trunc, index=idx, columns=[f"{g}__trunc" for g in gene_cols]),
        "stats": stats.astype(np.float32),
        "ratio": ratio.astype(np.float32),
    }
