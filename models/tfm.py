"""Tabular Foundation Model 래퍼. 범주형은 정수 코드(결측은 NaN)로 바꿔 numpy 로 전달."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class _TFM(BaseEstimator):
    def __init__(self, build, cat_cols, max_rows=None, seed=0):
        self.build, self.cat_cols = build, cat_cols
        self.max_rows, self.seed = max_rows, seed  # max_rows: 컨텍스트(학습) 행 상한, 넘으면 무작위 샘플

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].cat.codes.replace(-1, np.nan)
        return X.to_numpy(dtype=np.float32)

    def fit(self, X, y):
        if self.max_rows and len(X) > self.max_rows:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_rows, replace=False)
            X, y = X.iloc[idx], np.asarray(y)[idx]
        cat_idx = [X.columns.get_loc(c) for c in self.cat_cols]
        self.model_ = self.build(cat_idx)
        self.model_.fit(self._prep(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


def _tabpfn(task, cat_cols, seed, version=None):
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    cls = TabPFNRegressor if task == "regression" else TabPFNClassifier

    def build(cat_idx):
        kwargs = dict(categorical_features_indices=cat_idx, random_state=seed)
        if version is None:
            return cls(**kwargs)
        from tabpfn.constants import ModelVersion

        return cls.create_default_for_version(ModelVersion(version), **kwargs)

    return _TFM(build, cat_cols)


def make_tabpfn(task, cat_cols, seed):
    return _tabpfn(task, cat_cols, seed)


def make_tabpfn_v2(task, cat_cols, seed):
    return _tabpfn(task, cat_cols, seed, version="v2")


def make_tabicl(task, cat_cols, seed):
    from tabicl import TabICLClassifier, TabICLRegressor

    cls = TabICLRegressor if task == "regression" else TabICLClassifier
    return _TFM(lambda cat_idx: cls(random_state=seed), cat_cols)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _Transductive(BaseEstimator):
    """fit 에서 컨텍스트만 저장하고 predict 시점에 (train, test) 를 함께 넘기는 모델용 베이스.

    회귀 타깃은 표준화해서 넘기고 되돌림 (모델마다 내부 스케일 처리가 달라 안전하게 통일).
    """

    def __init__(self, task, cat_idx):
        self.task, self.cat_idx = task, cat_idx

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        if self.task == "regression":
            self.mu_, self.sd_ = float(np.mean(y)), float(np.std(y)) or 1.0
            self.y_ = (y - self.mu_) / self.sd_
        return self

    def _run(self, X):  # 분류: (n, n_classes) 확률, 회귀: (n,) 표준화 스케일 예측
        raise NotImplementedError

    def predict(self, X):
        out = self._run(X)
        return out * self.sd_ + self.mu_ if self.task == "regression" else out.argmax(1)

    def predict_proba(self, X):
        return self._run(X)


class _LimiX(_Transductive):
    def __init__(self, task, cat_idx, repo, filename, v2):
        super().__init__(task, cat_idx)
        self.repo, self.filename, self.v2 = repo, filename, v2

    def _run(self, X):
        import os

        import config as limix_config  # LimiX 패키지에 포함된 설정 JSON 디렉터리
        import torch
        from huggingface_hub import hf_hub_download
        from limix import LimiXPredictor

        for k, v in {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
                     "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}.items():
            os.environ.setdefault(k, v)
        kind = "reg" if self.task == "regression" else "cls"
        cfg = os.path.join(os.path.dirname(limix_config.__file__),
                           f"{kind}_default_noretrieval{'_v2' if self.v2 else ''}.json")
        model = LimiXPredictor(device=torch.device(_device()), model_path=hf_hub_download(self.repo, self.filename),
                               inference_config=cfg, categorical_features_indices=self.cat_idx)
        task_type = "Regression" if self.task == "regression" else "Classification"
        return np.asarray(model.predict(self.X_, self.y_, X, task_type=task_type))


class _Kumo(_Transductive):
    def __init__(self, task, cat_idx, n_estimators=8):
        super().__init__(task, cat_idx)
        self.n_estimators = n_estimators

    def _run(self, X):
        import sdm
        import torch

        dev = torch.device(_device())
        cols = [f"f{i}" for i in range(X.shape[1])]
        stypes = {c: "numerical" for c in cols}  # Kumo 는 수치형 피처만 지원 → 범주형은 정수 코드
        ctx = pd.DataFrame(self.X_, columns=cols).assign(target=self.y_)
        target_stype = "numerical" if self.task == "regression" else "categorical"
        ctx = sdm.TableTensor.from_pandas(ctx, stypes={**stypes, "target": target_stype}, device=dev)
        query = sdm.TableTensor.from_pandas(pd.DataFrame(X, columns=cols), stypes=stypes, device=dev)
        model = sdm.models.KumoTabular(task="regression" if self.task == "regression" else "classification",
                                       device=dev)
        with torch.amp.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x_context=ctx.drop_columns("target"), y_context=ctx[:, "target"], x_query=query,
                        num_estimators=self.n_estimators)
        out = out.to_pandas() if hasattr(out, "to_pandas") else pd.DataFrame(out.float().cpu().numpy())
        if self.task == "regression":  # 999개 분위수(q001~q999) 평균 ≈ 기대값
            return out.to_numpy(dtype=float).mean(1)
        return out[sorted(out.columns, key=lambda c: int(c) if str(c).isdigit() else str(c))].to_numpy(float)


def make_limix2(task, cat_cols, seed):
    return _TFM(lambda cat_idx: _LimiX(task, cat_idx, "stable-ai/LimiX-2", "LimiX-2.ckpt", v2=True), cat_cols)


def make_limix2m(task, cat_cols, seed):
    return _TFM(lambda cat_idx: _LimiX(task, cat_idx, "stable-ai/LimiX-2M", "LimiX-2M.ckpt", v2=False), cat_cols)


def make_kumo(task, cat_cols, seed):
    return _TFM(lambda cat_idx: _Kumo(task, cat_idx), cat_cols)


def _patch_exaone_sdpa():
    """EXAONE 는 flash attention 을 고정 선택하는데 T4 등 sm80 미만 GPU 에는 그 커널이 없음 → math 커널로 대체."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (8, 0):
        return
    from exaonetabular.model import attention
    from torch.nn.attention import SDPBackend

    attention._select_sdpa_backend = lambda *a, **k: SDPBackend.MATH


def make_exaone(task, cat_cols, seed):
    from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor

    _patch_exaone_sdpa()

    cls = EXAONETabularRegressor if task == "regression" else EXAONETabularClassifier
    return _TFM(lambda cat_idx: cls.from_pretrained(device=_device()), cat_cols)


def _mitra(task, cat_cols, seed, fine_tune):
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier, MitraRegressor
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if task == "regression":
        # mitra-regressor-2 는 1000-분위수 출력이라 AutoGluon 1.6 sklearn 인터페이스와 맞지 않음 → v1 사용
        cls, repo = MitraRegressor, "autogluon/mitra-regressor"
    else:
        cls, repo = MitraClassifier, "autogluon/mitra-classifier-2"

    # sklearn 인터페이스는 AutoGluon 전처리를 거치지 않으므로 결측만 직접 채움
    def build(cat_idx):
        model = cls(hf_model=repo, fine_tune=fine_tune, device=_device(), seed=seed, verbose=False)
        if task == "regression":
            model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), model)

    return _TFM(build, cat_cols)


def make_mitra(task, cat_cols, seed):
    return _mitra(task, cat_cols, seed, fine_tune=False)


def make_mitra_ft(task, cat_cols, seed):
    return _mitra(task, cat_cols, seed, fine_tune=True)  # 라이브러리 기본: 50 step fine-tuning
