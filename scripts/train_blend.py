"""재현 스크립트: 모델 학습 → 확률 섞기 → 제출 파일 저장을 한 번에 실행합니다.

LB에 실제로 낸 파일과 최종 후보를 같은 설정으로 다시 만듭니다.
- 프리셋에 넣는 기준: **LB에 실제로 낸 파일 + 최종 후보만**
  - blend14 (14, LB 0.3247): 최종 후보
  - select6 (16, LB 0.3183): 기록용, 최종 후보 아님
  - 17 XGB 시드 평균은 프리셋이 아니라 --xgb-seeds 옵션 (LB 미확인)
- 피처·모델 설정은 14·16과 동일 (src/의 공통 함수 사용)
  - LR: any / LR_trunc: any + trunc / XGB(10 설정)·RF·ET: any + stats
- 모든 모델은 팀 공통 fold(seed 42)로 CV → OOF 확률과 test 확률(fold 5개 평균)을 저장
- 섞는 비율은 노트북에서 정한 고정 비율 그대로 (여기서 다시 고르지 않음)
- 모델별 확률은 submissions/cache/에 저장 → 다시 실행하면 학습 없이 섞기만 함
  - 캐시 이름에 설정 해시를 붙임 (모델 파라미터, 피처, 시드, src/ 코드, 라이브러리 버전, CPU 종류)
    → src/data.py나 모델 설정이 바뀌면 자동으로 새로 학습
  - **최종 제출 파일은 항상 --no-cache로 생성** (해시가 못 잡는 변경까지 확실히 반영)

실행 (저장소 루트에서, --preset 필수)
------------------------------------
    python scripts/train_blend.py --list                       # 프리셋 목록
    python scripts/train_blend.py --preset blend14             # 14: XGB 0.7 + LR 0.3 (현재 최고 LB 0.3247)
    python scripts/train_blend.py --preset blend14 --xgb-seeds 42 0 1 2 3   # 17: XGB 시드 5개 평균 (LB 미확인)
    python scripts/train_blend.py --preset select6             # 16: 기록용 (LB 0.3183, 최종 후보 아님)
    python scripts/train_blend.py --preset blend14 --no-cache  # 최종 제출용 (캐시 무시)

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
  - OOF로 비율을 고르는 방법(16 Caruana)도 컴퓨터에 따라 다른 비율이 나옴 (select6 주석 참고)
- 팀 규칙 (비율을 고른 컴퓨터와 파일을 만든 컴퓨터가 다를 수 있어서 구분)
  - **비율**: 어느 컴퓨터에서 골랐는지 적고 고정 (PRESETS 주석)
  - **제출 파일**: 항상 설아님 Windows 데스크탑에서 생성 (0.3247을 만든 환경)
  - 다른 컴퓨터에서 만든 파일은 LB 차이가 모델 효과인지 환경 차이인지 구분할 수 없음
- 실행할 때마다 OS·CPU·라이브러리 버전을 맨 앞에 출력함 → 결과를 공유할 때 같이 기록
- 캐시는 만든 컴퓨터의 결과 → 캐시 해시에 CPU 종류가 들어 있어 다른 컴퓨터에서는 자동으로 새로 학습
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEED, SUBMISSION_DIR  # noqa: E402
from src.data import get_gene_cols, load_data  # noqa: E402
from src.features import mutation_features  # noqa: E402
from src.validation import encode_target, get_folds, macro_f1, run_cv, save_submission, seed_everything  # noqa: E402

# ===== 섞기 프리셋: LB에 실제로 낸 파일 + 최종 후보만 =====
PRESETS = {
    # 14 CHOSEN = {'XGB': 0.7, 'LR': 0.3}, OOF 0.4231 → LB 0.3247 (현재 최고, 최종 후보)
    #   비율: 14에서 고정 (XGB+LR 비율 0.2~0.5 중 OOF 최고) / 파일: 설아님 데스크탑
    "blend14": {"XGB": 0.7, "LR": 0.3},
    # 16 앙상블 선택 (핫스팟 제외) → LB 0.3183, 최종 후보 아님 (기록용)
    #   비율·파일 모두 설아님 데스크탑: W_NOHS 출력 {RF 0.36, LR_trunc 0.29, XGB 0.21, ET 0.14}
    #     = Caruana 선택 횟수 5:4:3:2 (÷14) (출력은 소수 둘째 자리, 20회 이하 선택 중 이 반올림이 나오는 조합은 이것뿐)
    #   참고: 선영님 컴퓨터 비율은 LR_trunc 5 / RF 4 / XGB 3 / ET 1 (÷13) → 같은 노트북도 컴퓨터마다 비율이 다름
    "select6": {"RF": 5 / 14, "LR_trunc": 4 / 14, "XGB": 3 / 14, "ET": 2 / 14},
}

# ===== 모델 정의 (14·16과 동일): 이름 → (피처 이름, 모델 만드는 함수(seed)) =====
XGB10 = dict(n_estimators=200, learning_rate=0.1, max_depth=3, colsample_bytree=0.3,
             tree_method="hist", eval_metric="mlogloss", n_jobs=-1)
TREE_KW = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
               class_weight="balanced_subsample", n_jobs=-1)


class BalancedXGB(xgb.XGBClassifier):
    """fold 학습 부분 기준 balanced sample_weight를 넣어 학습 (09~14와 동일)."""

    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kw)


MODELS = {
    "XGB": ("any+stats", lambda s: BalancedXGB(**XGB10, random_state=s)),
    "LR": ("any", lambda s: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=s)),
    "LR_trunc": ("any+trunc", lambda s: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=s)),
    "RF": ("any+stats", lambda s: RandomForestClassifier(**TREE_KW, random_state=s)),
    "ET": ("any+stats", lambda s: ExtraTreesClassifier(**TREE_KW, random_state=s)),
}
SEEDED = {"XGB"}   # --xgb-seeds가 적용되는 모델 (LR은 무작위성 없음, RF·ET는 500그루라 시드 영향 작음)


# ===== 피처 =====
def build_features(train, test, genes, needed):
    """필요한 피처 세트만 만듦 (14·16과 동일). 반환: {이름: (train X, test X)}"""
    F_tr, F_te = mutation_features(train, genes), mutation_features(test, genes)
    feats = {}
    if "any" in needed:
        feats["any"] = (F_tr["any"].astype("float32"), F_te["any"].astype("float32"))
    if "any+trunc" in needed:
        feats["any+trunc"] = (pd.concat([F_tr["any"], F_tr["trunc"]], axis=1),
                              pd.concat([F_te["any"], F_te["trunc"]], axis=1))
    if "any+stats" in needed:
        feats["any+stats"] = (pd.concat([F_tr["any"], F_tr["stats"]], axis=1),
                              pd.concat([F_te["any"], F_te["stats"]], axis=1))
    return feats


# ===== 학습 (캐시 사용) =====
_SRC_FILES = ["src/config.py", "src/data.py", "src/features.py", "src/validation.py"]


def config_hash(name, seed):
    """캐시 이름에 붙일 설정 해시. 아래 중 하나라도 바뀌면 다른 캐시 → 새로 학습.
    모델 파라미터, 피처 이름, 시드, src/ 코드(전처리·피처·CV), 라이브러리 버전, CPU 종류"""
    feat, make = MODELS[name]
    info = {
        "model": name, "feat": feat, "seed": seed,
        "params": repr(make(seed).get_params()),
        "src": [hashlib.md5((ROOT / f).read_bytes()).hexdigest() for f in _SRC_FILES],
        "versions": [xgb.__version__, sklearn.__version__, np.__version__, pd.__version__],
        "cpu": platform.machine(),
    }
    return hashlib.md5(json.dumps(info, sort_keys=True).encode()).hexdigest()[:8]


def get_proba(name, seed, feats, y, folds, cache_dir, use_cache):
    """모델 1개(시드 1개)의 OOF·test 확률. 같은 설정 해시의 캐시가 있으면 불러옴."""
    path = cache_dir / f"{name}_s{seed}_{config_hash(name, seed)}.npz"
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
    ap.add_argument("--preset", choices=list(PRESETS), help="섞기 조합 (필수, --list로 목록 확인)")
    ap.add_argument("--xgb-seeds", type=int, nargs="+", default=[SEED], help="XGB 시드 목록 (여러 개면 확률 평균, 17)")
    ap.add_argument("--name", default=None, help="제출 파일 이름 (기본: blend_<preset>[_XGB<n>seed])")
    ap.add_argument("--no-cache", action="store_true", help="캐시를 무시하고 새로 학습")
    ap.add_argument("--list", action="store_true", help="프리셋 목록 출력")
    args = ap.parse_args()

    if args.list:
        for k, w in PRESETS.items():
            print(f"{k:10s}", {m: round(v, 3) for m, v in w.items()})
        return
    if args.preset is None:
        ap.error("--preset을 지정하세요 (최종 조합 확정 전이라 기본값 없음). 목록: --list")
    weights = PRESETS[args.preset]
    assert abs(sum(weights.values()) - 1) < 1e-9, "비율 합이 1이 아닙니다."

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
            print(f"  {name:4s} seed {s:<3d} CV {cv:.4f} ± {sd:.4f}  ({how}, {time.time() - t:.0f}s)")
        P_oof[name], P_test[name] = np.mean(oofs, axis=0), np.mean(tests, axis=0)
        if len(seeds) > 1:
            cv, sd = fold_cv(P_oof[name], y, folds)
            print(f"  {name:4s} 시드 {len(seeds)}개 평균 CV {cv:.4f} ± {sd:.4f}")

    oof = sum(w * P_oof[m] for m, w in weights.items())
    tst = sum(w * P_test[m] for m, w in weights.items())
    cv, sd = fold_cv(oof, y, folds)
    oof_score = macro_f1(y, oof.argmax(1))
    print(f"섞기 CV {cv:.4f} ± {sd:.4f} | OOF {oof_score:.4f}")
    if args.preset == "select6":
        print("  ※ select6 (LB 0.3183, 최종 후보 아님). 비율을 같은 OOF로 고른 조합이라 이 OOF는 낙관적 (16 교차 적합 0.4261)")
    if not args.no_cache:
        print("  ※ 캐시 사용 실행. 최종 제출 파일은 --no-cache로 다시 생성하세요")

    n_seed = len(args.xgb_seeds)
    name = args.name or f"blend_{args.preset}" + (f"_XGB{n_seed}seed" if n_seed > 1 and SEEDED & set(weights) else "")
    path = save_submission(submission, tst, le, name=name, cv_score=oof_score)   # 파일명 숫자 = OOF (14와 같은 기준)

    s, sample = pd.read_csv(path), pd.read_csv(ROOT / "data" / "sample_submission.csv")
    assert s.shape == sample.shape and (s["ID"] == sample["ID"]).all(), "제출 형식이 sample_submission과 다릅니다."
    top = s["SUBCLASS"].value_counts(normalize=True).head(4).round(3).to_dict()
    print(f"형식 확인 OK {s.shape} | 예측 상위 {top} | 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
