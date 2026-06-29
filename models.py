"""
models.py
=========
Model zoo for the multi-disease prediction study.

Includes:
  * Classical baselines : LogisticRegression, RandomForest, XGBoost
  * MLP                 : simple tabular neural net (PyTorch)
  * FTTransformer       : Feature-Tokenizer Transformer (Gorishniy et al., 2021),
                          implemented from scratch (no fragile dependency)
  * TabNetAdapter       : wraps pytorch-tabnet if installed; otherwise raises a
                          clear message so the rest of the pipeline still runs
  * StackingEnsemble    : out-of-fold stacking meta-learner over base models

All neural models expose a scikit-style fit(X, y) / predict_proba(X) returning
probabilities for the positive class, so they slot into the same evaluation code.

These are REAL, runnable models. You train them on the real downloaded data in
Colab to obtain the numbers the reviewers asked for.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Classical baselines
# ============================================================================
def make_logreg():
    return LogisticRegression(max_iter=2000, class_weight="balanced")

def make_rf():
    return RandomForestClassifier(n_estimators=400, max_depth=None,
                                  class_weight="balanced", n_jobs=-1,
                                  random_state=RANDOM_SEED)

def make_xgb():
    if not _HAS_XGB:
        raise ImportError("xgboost not installed")
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1)


# ============================================================================
# Shared training utilities for neural models
# ============================================================================
def _to_tensor(X):
    if hasattr(X, "values"):
        X = X.values
    return torch.tensor(np.asarray(X, dtype=np.float32))

def _train_torch(model, X, y, epochs=100, lr=1e-3, batch=64, patience=10,
                 weight_decay=1e-4, verbose=False):
    model = model.to(DEVICE)
    Xt = _to_tensor(X).to(DEVICE)
    yt = torch.tensor(np.asarray(y, dtype=np.float32)).to(DEVICE)
    # class imbalance handling via pos_weight
    pos = float(yt.sum()); neg = float(len(yt) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(Xt)
    idx = np.arange(n)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx)
        epoch_loss = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            logit = model(Xt[b]).squeeze(-1)
            loss = loss_fn(logit, yt[b])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(b)
        epoch_loss /= n
        # simple early stopping on training loss (val handled outside in CV)
        if epoch_loss < best - 1e-4:
            best, best_state, wait = epoch_loss, \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep:3d}  loss {epoch_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def _predict_proba_torch(model, X):
    model.eval()
    Xt = _to_tensor(X).to(DEVICE)
    p = torch.sigmoid(model(Xt).squeeze(-1)).cpu().numpy()
    return np.clip(p, 1e-6, 1 - 1e-6)


# ============================================================================
# MLP
# ============================================================================
class _MLPNet(nn.Module):
    def __init__(self, d_in, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h),
                       nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP:
    def __init__(self, hidden=(128, 64), dropout=0.3, **kw):
        self.hidden, self.dropout, self.kw, self.model = hidden, dropout, kw, None
    def fit(self, X, y):
        self.model = _MLPNet(X.shape[1], self.hidden, self.dropout)
        self.model = _train_torch(self.model, X, y, **self.kw)
        return self
    def predict_proba(self, X):
        p = _predict_proba_torch(self.model, X)
        return np.column_stack([1 - p, p])


# ============================================================================
# FT-Transformer (Feature Tokenizer + Transformer), from scratch
# Gorishniy, Rubachev, Khrulkov, Babenko (NeurIPS 2021).
# Numeric features are tokenized via per-feature linear projection; a CLS token
# is appended and its final embedding is used for classification.
# (Here all inputs are already numeric/one-hot, so we tokenize every column.)
# ============================================================================
class _FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        # per-feature linear tokenizer: each scalar -> d_token vector
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        # x: (B, n_features) -> tokens (B, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)         # (B, 1+n_features, d_token)
        h = self.encoder(seq)
        h_cls = self.norm(h[:, 0])                    # CLS embedding
        return self.head(h_cls)

class FTTransformer:
    def __init__(self, d_token=32, n_heads=4, n_layers=3, dropout=0.1, **kw):
        self.cfg = dict(d_token=d_token, n_heads=n_heads,
                        n_layers=n_layers, dropout=dropout)
        self.kw, self.model = kw, None
    def fit(self, X, y):
        self.model = _FTTransformer(X.shape[1], **self.cfg)
        self.model = _train_torch(self.model, X, y, **self.kw)
        return self
    def predict_proba(self, X):
        p = _predict_proba_torch(self.model, X)
        return np.column_stack([1 - p, p])


# ============================================================================
# TabNet adapter (uses pytorch-tabnet if available)
# ============================================================================
class TabNetAdapter:
    def __init__(self, **kw):
        self.kw = kw; self.model = None
    def fit(self, X, y):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except Exception as e:
            raise ImportError(
                "pytorch-tabnet not installed. In Colab run: "
                "!pip install pytorch-tabnet") from e
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        yv = np.asarray(y)
        self.model = TabNetClassifier(seed=RANDOM_SEED, verbose=0)
        self.model.fit(Xv, yv, max_epochs=100, patience=15,
                       batch_size=256, virtual_batch_size=128)
        return self
    def predict_proba(self, X):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        return self.model.predict_proba(Xv)


# ============================================================================
# Stacking ensemble with out-of-fold meta-features (leakage-safe)
# ============================================================================
class StackingEnsemble:
    """
    Base learners are trained with internal k-fold to produce OUT-OF-FOLD
    predictions, on which a logistic-regression meta-learner is trained.
    This avoids the optimistic bias of training the meta-learner on in-sample
    base predictions.
    """
    def __init__(self, base_factories: dict, n_inner=5, seed=RANDOM_SEED):
        self.base_factories = base_factories      # name -> callable returning model
        self.n_inner = n_inner
        self.seed = seed
        self.fitted_bases_ = {}                    # name -> model trained on full data
        self.meta_ = None
        self.base_names_ = list(base_factories.keys())

    def fit(self, X, y):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        yv = np.asarray(y)
        n = len(Xv)
        oof = np.zeros((n, len(self.base_names_)))
        skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed)
        # build OOF predictions per base learner
        for j, name in enumerate(self.base_names_):
            for tr, va in skf.split(Xv, yv):
                m = self.base_factories[name]()
                m.fit(Xv[tr], yv[tr])
                oof[va, j] = m.predict_proba(Xv[va])[:, 1]
        # meta-learner on OOF
        self.meta_ = LogisticRegression(max_iter=2000)
        self.meta_.fit(oof, yv)
        # refit each base on ALL data for inference
        for name in self.base_names_:
            m = self.base_factories[name]()
            m.fit(Xv, yv)
            self.fitted_bases_[name] = m
        return self

    def predict_proba(self, X):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        meta_in = np.column_stack(
            [self.fitted_bases_[name].predict_proba(Xv)[:, 1]
             for name in self.base_names_])
        p = self.meta_.predict_proba(meta_in)[:, 1]
        return np.column_stack([1 - p, p])


# sklearn wrappers so classical models share predict_proba(X)[:,1] interface
class SkWrap:
    def __init__(self, factory): self.factory = factory; self.model = None
    def fit(self, X, y):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        self.model = self.factory(); self.model.fit(Xv, np.asarray(y)); return self
    def predict_proba(self, X):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        return self.model.predict_proba(Xv)


def default_base_factories(include_xgb=True):
    f = {
        "logreg": lambda: SkWrap(make_logreg),
        "rf": lambda: SkWrap(make_rf),
        "mlp": lambda: MLP(epochs=80, verbose=False),
        "ftt": lambda: FTTransformer(epochs=80, verbose=False),
    }
    if include_xgb and _HAS_XGB:
        f["xgb"] = lambda: SkWrap(make_xgb)
    return f


if __name__ == "__main__":
    # smoke test on synthetic data
    rng = np.random.default_rng(0)
    n, d = 300, 10
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = (1 / (1 + np.exp(-(X @ w))) > 0.5).astype(int)
    from sklearn.metrics import roc_auc_score

    for name, mk in [("logreg", lambda: SkWrap(make_logreg)),
                     ("rf", lambda: SkWrap(make_rf)),
                     ("mlp", lambda: MLP(epochs=40)),
                     ("ftt", lambda: FTTransformer(epochs=40))]:
        m = mk().fit(X[:240], y[:240])
        p = m.predict_proba(X[240:])[:, 1]
        print(f"{name:8s} AUC={roc_auc_score(y[240:], p):.3f}")

    ens = StackingEnsemble(default_base_factories(include_xgb=_HAS_XGB), n_inner=3)
    ens.fit(X[:240], y[:240])
    p = ens.predict_proba(X[240:])[:, 1]
    print(f"stack    AUC={roc_auc_score(y[240:], p):.3f}")
    print("models self-test OK")
