"""재현 스크립트: 모델 학습 → 확률 섞기 → 제출 파일 저장을 한 번에 실행합니다.

노트북(14, 16, 17)에서 고른 조합을 같은 설정으로 다시 만듭니다.
- 피처·모델 설정은 각 노트북과 동일 (src/의 공통 함수 사용)
- 모든 모델은 팀 공통 fold(seed 42)로 CV → OOF 확률과 test 확률(fold 5개 평균)을 저장
- 섞는 비율은 노트북에서 OOF로 고른 값을 그대로 고정 (여기서 다시 고르지 않음)
- 모델별 확률은 submissions/cache/에 저장 → 다시 실행하면 학습 없이 섞기만 함 (--no-cache로 새로 학습)

실행 (저장소 루트에서)
---------------------
    python scripts/train_blend.py --preset select6             # 16 최종 (핫스팟 제외)
    python scripts/train_blend.py --preset blend14             # 14: XGB 0.7 + LR 0.3
    python scripts/train_blend.py --preset blend14 --xgb-seeds 42 0 1 2 3   # 17: XGB 시드 5개 평균
    python scripts/train_blend.py --list                       # 프리셋 목록

새 조합 추가: PRESETS에 {모델 이름: 비율}을 추가 (모델은 MODELS, 피처는 build_features 참고)

실행 환경과 CV 차이 (17 리뷰에서 확인)
------------------------------------
- 라이브러리 버전이 같아도(requirements.txt) **CPU 종류에 따라 XGB 결과가 조금 다름**
  - XGBoost `hist`의 실수 합산 순서가 CPU마다 달라 분기가 조금씩 바뀜 (같은 컴퓨터에서는 반복 실행해도 결과 동일)
  - 예: XGB 10 설정, 정규화 데이터, 시드 42
      Windows (x86)          CV 0.4161 ± 0.0060  ← LB 0.3247(14)을 만든 환경
      macOS Apple M1 (arm64) CV 0.4125 ± 0.0109
  - 차이(약 0.004)가 시드를 바꿨을 때의 차이(0.4107 ~ 0.4145)와 비슷한 크기
  - LR(무작위성 없음)은 환경과 무관하게 같음 (0.3818). RF·ET는 ±0.001 수준
- 팀 규칙: **LB 비교용 제출 파일은 한 컴퓨터(0.3247을 만든 Windows 데스크탑)에서만 만듦**
  - 다른 컴퓨터에서 만든 파일은 LB 차이가 모델 효과인지 환경 차이인지 구분할 수 없음
- 실행할 때마다 OS·CPU·라이브러리 버전을 맨 앞에 출력함 → 결과를 공유할 때 같이 기록
- 캐시(submissions/cache/)는 만든 컴퓨터의 결과 → 다른 컴퓨터로 옮겨 쓰지 말고, 옮겼다면 --no-cache로 새로 학습
"""
import argparse
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEED, SUBMISSION_DIR, WT  # noqa: E402
from src.data import get_gene_cols, load_data  # noqa: E402
from src.features import mutation_features  # noqa: E402
from src.validation import encode_target, get_folds, macro_f1, run_cv, save_submission, seed_everything  # noqa: E402

# ===== 섞기 프리셋 (비율은 각 노트북에서 OOF로 고른 값) =====
PRESETS = {
    # 14: XGB 0.7 + LR 0.3 → LB 0.3247
    "blend14": {"XGB": 0.7, "LR": 0.3},
    # 16: 앙상블 선택, 핫스팟 제외 (Caruana 13회 선택: LR_trunc 5, RF 4, XGB 3, ET 1)
    "select6": {"LR_trunc": 5 / 13, "RF": 4 / 13, "XGB": 3 / 13, "ET": 1 / 13},
    # 16: 앙상블 선택, 후보 7개 (Caruana 20회 선택: XGB_hs 7, LR_trunc 7, RF 5, ET 1)
    "select7": {"XGB_hs": 7 / 20, "LR_trunc": 7 / 20, "RF": 5 / 20, "ET": 1 / 20},
}

# ===== 모델 정의: 이름 → (피처 이름, 모델 만드는 함수(seed)) =====
XGB10 = dict(n_estimators=200, learning_rate=0.1, max_depth=3, colsample_bytree=0.3,
             tree_method="hist", eval_metric="mlogloss", n_jobs=-1)
TREE_KW = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
               class_weight="balanced_subsample", n_jobs=-1)


class BalancedXGB(xgb.XGBClassifier):
    """fold 학습 부분 기준 balanced sample_weight를 넣어 학습 (09~16과 동일)."""

    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kw)


class FrequentTokens(BaseEstimator, TransformerMixin):
    """fold 학습 부분에서 min_count명 미만인 핫스팟 열(hs__)을 버림 (11, 16과 동일)."""

    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, X, y=None):
        hs = [c for c in X.columns if c.startswith("hs__")]
        n = X[hs].sum()
        self.drop_ = n.index[n < self.min_count].tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_)


def _lr(seed):
    return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=seed)


MODELS = {
    "XGB": ("any+stats", lambda s: BalancedXGB(**XGB10, random_state=s)),
    "XGB_hs": ("any+stats+hs", lambda s: make_pipeline(FrequentTokens(10), BalancedXGB(**XGB10, random_state=s))),
    "LR": ("any", _lr),
    "LR_trunc": ("any+trunc", _lr),
    "RF": ("any+stats", lambda s: RandomForestClassifier(**TREE_KW, random_state=s)),
    "ET": ("any+stats", lambda s: ExtraTreesClassifier(**TREE_KW, random_state=s)),
}
SEEDED = {"XGB", "XGB_hs"}   # --xgb-seeds가 적용되는 모델 (LR은 무작위성 없음, RF·ET는 500그루라 시드 영향 작음)


# ===== 피처 =====
def _hotspot_matrices(genes, index_tr, index_te):
    """'유전자_변이' 핫스팟 후보 행렬 (11, 16과 동일). 토큰은 dedupe=False 데이터로 만듦."""
    train_t, test_t, _ = load_data(dedupe=False)

    def tokens(df):
        out = []
        for row in df[genes].to_numpy():
            s = set()
            for g, v in zip(genes, row):
                if v != WT:
                    s.update(f"{g}_{t}" for t in v.split())
            out.append(s)
        return out

    tok_tr, tok_te = tokens(train_t), tokens(test_t)
    cnt = Counter(t for s in tok_tr for t in s)
    vocab = sorted(t for t, n in cnt.items() if n >= 5)   # 후보만 줄임 (실제 선택은 fold 안 FrequentTokens)
    col = {t: j for j, t in enumerate(vocab)}
    names = ["hs__" + t.replace("*", "stop").replace(">", "_to_") for t in vocab]

    def matrix(toks, index):
        M = np.zeros((len(toks), len(vocab)), np.int8)
        for i, s in enumerate(toks):
            for t in s:
                if t in col:
                    M[i, col[t]] = 1
        return pd.DataFrame(M, index=index, columns=names)

    return matrix(tok_tr, index_tr), matrix(tok_te, index_te)


def build_features(train, test, genes, needed):
    """필요한 피처 세트만 만듦. 반환: {이름: (train X, test X)}"""
    F_tr, F_te = mutation_features(train, genes), mutation_features(test, genes)
    cat = lambda parts: (pd.concat([F_tr[p] for p in parts], axis=1), pd.concat([F_te[p] for p in parts], axis=1))
    feats = {}
    if "any" in needed:
        feats["any"] = (F_tr["any"].astype("float32"), F_te["any"].astype("float32"))
    if "any+trunc" in needed:
        feats["any+trunc"] = cat(["any", "trunc"])
    if "any+stats" in needed or "any+stats+hs" in needed:
        feats["any+stats"] = cat(["any", "stats"])
    if "any+stats+hs" in needed:
        H_tr, H_te = _hotspot_matrices(genes, train.index, test.index)
        feats["any+stats+hs"] = (pd.concat([feats["any+stats"][0], H_tr], axis=1),
                                 pd.concat([feats["any+stats"][1], H_te], axis=1))
    return feats


# ===== 학습 (캐시 사용) =====
def get_proba(name, seed, feats, y, folds, cache_dir, use_cache):
    """모델 1개(시드 1개)의 OOF·test 확률. 캐시가 있으면 불러옴."""
    path = cache_dir / f"{name}_s{seed}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return z["oof"], z["test"], "cache"
    feat, make = MODELS[name]
    X, X_te = feats[feat]
    r = run_cv(lambda: make(seed), X, y, X_te, folds=folds, verbose=False)
    np.savez(path, oof=r["oof_proba"], test=r["test_proba"])
    return r["oof_proba"], r["test_proba"], "trained"


def fold_cv(P, y, folds):
    s = np.array([macro_f1(y[va], P[va].argmax(1)) for _, va in folds])
    return s.mean(), s.std()


def main():
    ap = argparse.ArgumentParser(description="학습 → 섞기 → 제출 파일 저장")
    ap.add_argument("--preset", default="select6", help=f"섞기 조합: {', '.join(PRESETS)}")
    ap.add_argument("--xgb-seeds", type=int, nargs="+", default=[SEED], help="XGB 시드 목록 (여러 개면 확률 평균, 17)")
    ap.add_argument("--name", default=None, help="제출 파일 이름 (기본: blend_<preset>[_XGB<n>seed])")
    ap.add_argument("--no-cache", action="store_true", help="캐시를 무시하고 새로 학습")
    ap.add_argument("--list", action="store_true", help="프리셋 목록 출력")
    args = ap.parse_args()

    if args.list:
        for k, w in PRESETS.items():
            print(f"{k:10s}", {m: round(v, 3) for m, v in w.items()})
        return
    weights = PRESETS[args.preset]
    assert abs(sum(weights.values()) - 1) < 1e-9, "비율 합이 1이 아닙니다."

    import sklearn
    print(f"[환경] {platform.system()} {platform.release()} | CPU {platform.machine()} ({platform.processor() or '-'}) | "
          f"python {platform.python_version()} | xgboost {xgb.__version__} | scikit-learn {sklearn.__version__} | "
          f"numpy {np.__version__} | pandas {pd.__version__}")
    print("       XGB 결과는 CPU 종류에 따라 CV ±0.004 정도 달라질 수 있음 → LB 비교용 파일은 지정된 한 컴퓨터에서만 생성")

    seed_everything()
    t0 = time.time()
    train, test, submission = load_data()   # 정규화 + 이소폼 중복 제거 (기본값)
    genes = get_gene_cols(train)
    y, le = encode_target(train)
    folds = get_folds(y)
    feats = build_features(train, test, genes, {MODELS[m][0] for m in weights})
    cache_dir = SUBMISSION_DIR / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    P_oof, P_test = {}, {}
    print(f"[{args.preset}] 비율: { {m: round(v, 3) for m, v in weights.items()} }")
    for name in weights:
        seeds = args.xgb_seeds if name in SEEDED else [SEED]
        oofs, tests = [], []
        for s in seeds:
            t = time.time()
            oof, tst, how = get_proba(name, s, feats, y, folds, cache_dir, not args.no_cache)
            oofs.append(oof); tests.append(tst)
            cv, sd = fold_cv(oof, y, folds)
            print(f"  {name:9s} seed {s:<3d} CV {cv:.4f} ± {sd:.4f}  ({how}, {time.time() - t:.0f}s)")
        P_oof[name], P_test[name] = np.mean(oofs, axis=0), np.mean(tests, axis=0)
        if len(seeds) > 1:
            cv, sd = fold_cv(P_oof[name], y, folds)
            print(f"  {name:9s} 시드 {len(seeds)}개 평균 CV {cv:.4f} ± {sd:.4f}")

    oof = sum(w * P_oof[m] for m, w in weights.items())
    tst = sum(w * P_test[m] for m, w in weights.items())
    cv, sd = fold_cv(oof, y, folds)
    oof_score = macro_f1(y, oof.argmax(1))
    print(f"섞기 CV {cv:.4f} ± {sd:.4f} | OOF {oof_score:.4f}")

    n_seed = len(args.xgb_seeds)
    name = args.name or f"blend_{args.preset}" + (f"_XGB{n_seed}seed" if n_seed > 1 and SEEDED & set(weights) else "")
    path = save_submission(submission, tst, le, name=name, cv_score=oof_score)   # 파일명 숫자 = OOF (14와 같은 기준)

    s, sample = pd.read_csv(path), pd.read_csv(ROOT / "data" / "sample_submission.csv")
    assert s.shape == sample.shape and (s["ID"] == sample["ID"]).all(), "제출 형식이 sample_submission과 다릅니다."
    top = s["SUBCLASS"].value_counts(normalize=True).head(4).round(3).to_dict()
    print(f"형식 확인 OK {s.shape} | 예측 상위 {top} | 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
