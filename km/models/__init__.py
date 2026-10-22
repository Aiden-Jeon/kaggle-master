"""모델 레지스트리. 모든 모델은 sklearn 스타일 fit / predict(_proba) 를 가짐.

새 모델 추가: 아래 REGISTRY 에 `이름: (모듈, 팩토리 함수명)` 등록, Kaggle 설치가 필요하면 KAGGLE_PIP 에도 추가.
팩토리 시그니처: make(task: str, cat_cols: list[str], seed: int, **설정) -> estimator (설정은 VARIANTS 에서 전달)

이름 뒤에 `@<spec>` 을 붙이면 컨텍스트 선택 래퍼로 감쌈 (예: tabpfn_v2@knn1000, models/context.py 참고).
"""

from __future__ import annotations

import importlib

REGISTRY = {
    "lgbm": ("km.models.gbdt", "make_lgbm"),
    "xgb": ("km.models.gbdt", "make_xgb"),
    "catboost": ("km.models.gbdt", "make_catboost"),
    "tabpfn": ("km.models.tfm", "make_tabpfn"),  # 최신 기본 버전 (첫 실행 시 브라우저 라이선스 동의 필요)
    "tabpfn_v2": ("km.models.tfm", "make_tabpfn_v2"),  # 라이선스 동의 없이 사용 가능
    "tabpfn35": ("km.models.tfm", "make_tabpfn35"),  # TabPFN-3.5 명시 (라이선스 동의 필요)
    "tabicl": ("km.models.tfm", "make_tabicl"),
    "limix2m": ("km.models.tfm", "make_limix2m"),  # 2M 경량판, 비상업 라이선스
    "mitra": ("km.models.tfm", "make_mitra"),  # Mitra-v2 zero-shot
    "mitra_ft": ("km.models.tfm", "make_mitra_ft"),  # Mitra-v2 + 50 step fine-tuning
    "exaone": ("km.models.tfm", "make_exaone"),  # 가중치 비상업 라이선스
    "kumo": ("km.models.tfm", "make_kumo"),  # NVIDIA Kumo-Tabular (large)
    "causilo": ("km.models.tfm", "make_causilo"),  # Nums AI Causilo, 가중치 비상업 라이선스
}

# 실행 목록에서 뺀 모델. km lb 는 제출 설명에서 이름을 알아본 뒤 순위 집계·자동 제출에서 뺌
# limix2 (LimiX-2 400M): GiveMeSomeCredit fold 당 ~1,400s, S4E2 8시간. 1위였던 대회도 2위와 차이가 작았음
RETIRED = {"limix2"}

_LIMIX = [["--no-deps", "git+https://github.com/limix-ldm-ai/LimiX.git"], ["einops", "kditransform", "nvtx"],
          ["-U", "numba"]]  # Kaggle 기본 numba 는 numpy<2.3 만 지원 → 다른 패키지가 numpy 를 올리면 kditransform 이 깨짐

# Kaggle 노트북에서 모델별로 추가 설치할 pip 인자 목록 (항목마다 pip install 한 번)
KAGGLE_PIP = {
    "tabpfn": [["tabpfn"]],
    "tabpfn_v2": [["tabpfn"]],
    "tabpfn35": [["tabpfn"]],
    "tabicl": [["tabicl"]],
    "limix2m": _LIMIX,  # torch==2.9.1 고정을 피하려고 --no-deps 로 설치
    "mitra": [["autogluon.tabular[mitra]>=1.6"]],
    "mitra_ft": [["autogluon.tabular[mitra]>=1.6"]],
    "exaone": [["exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"]],
    "kumo": [["structured-data-models"]],
    "causilo": [["causilo"]],
}


# 앙상블 크기·피처 상한 벤치마크용 변형: 이름 → (REGISTRY 모델, 인자). 기본값을 정하면 정리
# 인자 중 max_features 는 대회 설정 tfm_max_features 대신 쓰고, 나머지는 팩토리로 전달
VARIANTS = {
    "kumo_e4": ("kumo", dict(n_estimators=4)),
}


def base_name(name: str) -> str:
    base = name.partition("@")[0]
    return VARIANTS.get(base, (base,))[0]


def make_model(name: str, task: str, cat_cols: list[str], seed: int, max_rows: int | None = None,
               max_features: int | None = None):
    """max_rows / max_features: TFM 컨텍스트 행·피처 상한 (대회 설정 tfm_max_rows / tfm_max_features).
    GBDT 와 `@spec` 래퍼(컨텍스트를 직접 고름)에는 적용 안 함."""
    base, _, spec = name.partition("@")
    base, kwargs = VARIANTS.get(base, (base, {}))
    kwargs = dict(kwargs)
    max_features = kwargs.pop("max_features", max_features)
    if base not in REGISTRY:
        raise KeyError(f"알 수 없는 모델 '{base}'. 사용 가능: {', '.join([*REGISTRY, *VARIANTS])}")
    module, fn = REGISTRY[base]
    factory = getattr(importlib.import_module(module), fn)
    if not spec:
        from km.models.tfm import _TFM

        model = factory(task, cat_cols, seed, **kwargs)
        if isinstance(model, _TFM):
            model.set_params(max_rows=max_rows, max_features=max_features, task=task)
        return model
    from km.models.context import wrap

    return wrap(lambda: factory(task, cat_cols, seed, **kwargs), spec, task, cat_cols, seed)
