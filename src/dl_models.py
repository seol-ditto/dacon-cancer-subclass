"""딥러닝 분류기 (17번 노트북용).

src/validation.py의 run_cv에 그대로 넣을 수 있도록 sklearn 형식(fit / predict_proba)을 따릅니다.

입력 X는 pd.concat([F['any'], F['stats']], axis=1) 순서라고 가정합니다.
    - 앞 n_any개 열: 유전자별 변이 여부 (0/1)
    - 나머지 열: stats (변이 유형별 개수, log)

누수 방지
    - stats 표준화, 클래스 가중치, 가짜 변이 증강용 유전자 빈도는 모두 fit()에 들어온
      학습 fold 데이터로만 계산합니다. test나 검증 fold 정보는 쓰지 않습니다.
    - 조기 종료는 학습 fold 안에서 다시 10%를 떼어 판단합니다.

모델 2종
    - MLPClassifierTorch      : 변이 여부 + stats를 그대로 넣는 2층 MLP
    - GeneSetTransformer      : 환자별 '변이된 유전자'만 토큰으로 넣고 어텐션으로 섞는 모델
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():  # 맥북 (Apple Silicon)
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# 공통 부분: 전처리, 증강, 학습 루프
# --------------------------------------------------------------------------
class _BaseTorchClassifier:
    """fit/predict_proba 공통 로직. 하위 클래스는 _build, _batch_logits만 구현합니다."""

    def __init__(self, n_any, n_classes=26, epochs=60, batch_size=128, lr=1e-3,
                 weight_decay=1e-4, patience=8, aug_prob=0.5, aug_max_ratio=1.0,
                 gene_drop=0.1, val_size=0.1, seed=42, device=None, verbose=False):
        self.n_any = n_any
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.aug_prob = aug_prob            # 배치 안에서 증강을 적용할 환자 비율
        self.aug_max_ratio = aug_max_ratio  # 가짜 변이 수 상한 = 원래 변이 수 × 이 값 (test가 약 2배라 1.0)
        self.gene_drop = gene_drop          # 진짜 변이 중 무작위로 지우는 비율
        self.val_size = val_size
        self.seed = seed
        self.device = device or get_device()
        self.verbose = verbose

    # ---- 전처리 (학습 fold로만 맞춤) ----
    def _split(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X[:, :self.n_any], X[:, self.n_any:]

    def _fit_prep(self, A, S):
        self.s_mean_ = S.mean(0)
        self.s_std_ = S.std(0) + 1e-6
        freq = A.astype(np.float64).mean(0) + 1e-8   # 학습 fold의 유전자별 변이 빈도
        self.gene_cdf_ = np.cumsum(freq / freq.sum())

    def _scale_stats(self, S):
        return (S - self.s_mean_) / self.s_std_

    # ---- 증강: 학습 fold 빈도로 가짜 변이 추가 + 진짜 변이 일부 삭제 ----
    def _augment(self, A, rng):
        A = A.copy()
        n = len(A)
        idx = np.where(rng.random(n) < self.aug_prob)[0]
        for i in idx:
            mut = np.flatnonzero(A[i])
            if self.gene_drop > 0 and len(mut) > 1:
                drop = mut[rng.random(len(mut)) < self.gene_drop]
                A[i, drop] = 0
            k = rng.integers(0, int(max(1, len(mut)) * self.aug_max_ratio) + 1)
            if k > 0:
                fake = np.searchsorted(self.gene_cdf_, rng.random(k) * self.gene_cdf_[-1])
                fake = np.minimum(fake, self.n_any - 1)
                A[i, fake] = 1
        return A

    # ---- 학습 ----
    def fit(self, X, y):
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        A, S = self._split(X)
        tr, va = train_test_split(np.arange(len(y)), test_size=self.val_size,
                                  stratify=y, random_state=self.seed)
        self._fit_prep(A[tr], S[tr])
        cw = compute_class_weight('balanced', classes=np.arange(self.n_classes), y=y[tr]) \
            if len(np.unique(y[tr])) == self.n_classes else None
        weight = torch.tensor(cw, dtype=torch.float32, device=self.device) if cw is not None else None

        self.model_ = self._build(S.shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        Sva = self._scale_stats(S[va]); yva = y[va]
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.model_.train()
            order = rng.permutation(tr)
            for b in range(0, len(order), self.batch_size):
                bi = order[b:b + self.batch_size]
                Ab = self._augment(A[bi], rng)
                logits = self._batch_logits(Ab, self._scale_stats(S[bi]), train=True, rng=rng)
                loss = Fn.cross_entropy(logits, torch.tensor(y[bi], device=self.device), weight=weight)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            # 검증 손실 (가중치 없는 logloss)로 조기 종료
            p = self._predict(A[va], Sva)
            vl = -np.log(np.clip(p[np.arange(len(yva)), yva], 1e-7, 1)).mean()
            if self.verbose:
                print(f'  epoch {ep+1:3d}  val_logloss {vl:.4f}')
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_val_logloss_ = best
        self.n_epochs_ = ep + 1
        return self

    @torch.no_grad()
    def _predict(self, A, S_scaled):
        self.model_.eval()
        out = []
        for b in range(0, len(A), 512):
            logits = self._batch_logits(A[b:b + 512], S_scaled[b:b + 512], train=False)
            out.append(torch.softmax(logits, 1).float().cpu().numpy())
        return np.vstack(out)

    def predict_proba(self, X):
        A, S = self._split(X)
        return self._predict(A, self._scale_stats(S))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# --------------------------------------------------------------------------
# 1) MLP
# --------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, n_any, n_stats, n_classes, hidden=(512, 128), dropout=0.5, in_drop=0.1):
        super().__init__()
        layers, d = [nn.Dropout(in_drop)], n_any + n_stats
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifierTorch(_BaseTorchClassifier):
    def __init__(self, n_any, hidden=(512, 128), dropout=0.5, in_drop=0.1, **kw):
        super().__init__(n_any, **kw)
        self.hidden, self.dropout, self.in_drop = hidden, dropout, in_drop

    def _build(self, n_stats):
        return _MLP(self.n_any, n_stats, self.n_classes, self.hidden, self.dropout, self.in_drop)

    def _batch_logits(self, A, S, train=False, rng=None):
        x = torch.tensor(np.hstack([A, S]), dtype=torch.float32, device=self.device)
        return self.model_(x)


# --------------------------------------------------------------------------
# 2) 변이 유전자 집합 트랜스포머
# --------------------------------------------------------------------------
class _GeneSetNet(nn.Module):
    def __init__(self, n_any, n_stats, n_classes, d=96, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_any + 2, d, padding_idx=0)  # 0=패딩, 1=CLS, 유전자 i → i+2
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 2, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.stats = nn.Sequential(nn.Linear(n_stats, d), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(d * 3), nn.Dropout(dropout), nn.Linear(d * 3, n_classes))

    def forward(self, tok, pad_mask, s):
        h = self.encoder(self.emb(tok), src_key_padding_mask=pad_mask)
        cls = h[:, 0]
        keep = (~pad_mask).unsqueeze(-1).float()
        mean = (h * keep).sum(1) / keep.sum(1).clamp(min=1)  # CLS 포함 평균 (변이 0개 환자도 안전)
        return self.head(torch.cat([cls, mean, self.stats(s)], 1))


class GeneSetTransformer(_BaseTorchClassifier):
    """환자별 변이 유전자 목록을 토큰으로 넣습니다.

    max_len보다 변이가 많으면
        - 학습: 매 배치 무작위로 max_len개를 고름
        - 예측: 무작위 선택을 tta번 반복해 확률 평균 (시드 고정이라 재현 가능)
    """

    def __init__(self, n_any, d=96, heads=4, layers=2, dropout=0.2, max_len=128, tta=4, **kw):
        super().__init__(n_any, **kw)
        self.d, self.heads, self.layers, self.drop = d, heads, layers, dropout
        self.max_len, self.tta = max_len, tta

    def _build(self, n_stats):
        return _GeneSetNet(self.n_any, n_stats, self.n_classes, self.d, self.heads, self.layers, self.drop)

    def _tokens(self, A, rng):
        rows = [np.flatnonzero(a) for a in A]
        rows = [r if len(r) <= self.max_len else np.sort(rng.choice(r, self.max_len, replace=False))
                for r in rows]
        L = 1 + max(len(r) for r in rows)
        tok = np.zeros((len(A), L), dtype=np.int64)
        tok[:, 0] = 1  # CLS
        for i, r in enumerate(rows):
            tok[i, 1:1 + len(r)] = r + 2
        tok = torch.tensor(tok, device=self.device)
        return tok, tok == 0

    def _batch_logits(self, A, S, train=False, rng=None):
        s = torch.tensor(S, dtype=torch.float32, device=self.device)
        if train:
            tok, mask = self._tokens(A, rng)
            return self.model_(tok, mask, s)
        need_tta = (A.sum(1) > self.max_len).any()
        n = self.tta if need_tta else 1
        r = np.random.default_rng(self.seed)
        logp = 0
        for _ in range(n):
            tok, mask = self._tokens(A, r)
            logp = logp + torch.log_softmax(self.model_(tok, mask, s), 1)
        return logp / n  # 로그 확률 평균 → softmax하면 기하 평균 확률
