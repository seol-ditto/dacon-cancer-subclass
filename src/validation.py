"""공통 검증 도구: 시드 고정, Stratified K-Fold, Macro F1, CV 실행, 제출 파일 저장.

팀원 간 CV 점수를 비교할 수 있도록 모두 이 모듈의 run_cv()로 검증합니다.

사용 예시
--------
>>> from sklearn.linear_model import LogisticRegression
>>> from src.data import load_data, get_gene_cols
>>> from src.features import to_binary
>>> from src.validation import seed_everything, encode_target, run_cv, save_submission
>>> seed_everything()
>>> train, test, submission = load_data()
>>> genes = get_gene_cols(train)
>>> X, X_test = to_binary(train, genes), to_binary(test, genes)
>>> y, le = encode_target(train)
>>> result = run_cv(lambda: LogisticRegression(max_iter=1000, class_weight="balanced"), X, y, X_test)
>>> save_submission(submission, result["test_proba"], le, name="eunji_lr_l0", cv_score=result["cv_mean"])
"""
import os
import random
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.config import N_FOLDS, SEED, SUBMISSION_DIR, TARGET_COL


def seed_everything(seed=SEED):
    """파이썬/넘파이 난수 시드를 고정합니다. (모델에는 random_state=SEED를 따로 지정하세요)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def encode_target(train):
    """SUBCLASS 문자열 → 0~25 정수. LabelEncoder는 train 라벨로만 fit합니다.

    Returns
    -------
    y : np.ndarray, le : LabelEncoder (제출 시 inverse_transform에 사용)
    """
    le = LabelEncoder()
    y = le.fit_transform(train[TARGET_COL])
    return y, le


def get_folds(y, n_splits=N_FOLDS, seed=SEED):
    """팀 공통 Stratified K-Fold 분할. 같은 y와 seed면 항상 같은 분할이 나옵니다.

    Returns
    -------
    list of (train_idx, valid_idx)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def macro_f1(y_true, y_pred):
    """대회 평가 지표 (Macro F1)."""
    return f1_score(y_true, y_pred, average="macro")


def _take(X, idx):
    """DataFrame / ndarray / scipy sparse 모두에서 행 선택."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def run_cv(make_model, X, y, X_test=None, folds=None, verbose=True):
    """공통 CV 실행.

    Parameters
    ----------
    make_model : 인자 없이 호출하면 '새' 모델을 돌려주는 함수 (fold마다 새로 만듦)
        예) lambda: LogisticRegression(max_iter=1000, random_state=SEED)
    X, y : 학습 피처, 정수 라벨
    X_test : 주어지면 fold별 모델의 test 예측 확률을 평균해 반환
    folds : get_folds() 결과. None이면 팀 공통 분할 사용

    Returns
    -------
    dict
        fold_scores : fold별 Macro F1
        cv_mean, cv_std : 평균, 표준편차 (실험 기록표에는 cv_mean 기록)
        oof_score : OOF 전체 Macro F1
        oof_proba : (n_train, n_class) OOF 예측 확률 → 후처리·앙상블에 사용
        test_proba : (n_test, n_class) fold 평균 예측 확률 또는 None
        models : fold별 학습된 모델
    """
    if folds is None:
        folds = get_folds(y)
    n_class = len(np.unique(y))

    oof_proba = np.zeros((len(y), n_class))
    test_proba = None if X_test is None else np.zeros((X_test.shape[0], n_class))
    fold_scores, models = [], []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = make_model()
        model.fit(_take(X, tr_idx), y[tr_idx])

        va_proba = model.predict_proba(_take(X, va_idx))
        oof_proba[va_idx] = va_proba
        score = macro_f1(y[va_idx], va_proba.argmax(axis=1))
        fold_scores.append(score)
        models.append(model)

        if X_test is not None:
            test_proba += model.predict_proba(X_test) / len(folds)
        if verbose:
            print(f"[fold {fold}] Macro F1 = {score:.4f}")

    result = {
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_score": macro_f1(y, oof_proba.argmax(axis=1)),
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "models": models,
    }
    if verbose:
        print(f"CV Macro F1 = {result['cv_mean']:.4f} ± {result['cv_std']:.4f} "
              f"(OOF {result['oof_score']:.4f})")
    return result


def save_submission(submission, test_proba, le, name, cv_score):
    """제출 파일 저장: submissions/날짜_이름_모델_CV점수.csv (docs/team_rules.md 형식).

    name 예시: 'eunji_lr_l0'  →  1109_eunji_lr_l0_0.512.csv
    """
    sub = submission.copy()
    sub[TARGET_COL] = le.inverse_transform(np.asarray(test_proba).argmax(axis=1))
    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"{datetime.now():%m%d}_{name}_{cv_score:.3f}.csv"
    sub.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {path}")
    return path
