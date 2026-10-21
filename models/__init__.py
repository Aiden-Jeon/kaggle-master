"""모델 레지스트리. 모든 모델은 sklearn 스타일 fit / predict(_proba) 를 가짐.

새 모델 추가: 아래 REGISTRY 에 `이름: (모듈, 팩토리 함수명)` 등록, Kaggle 설치가 필요하면 KAGGLE_PIP 에도 추가.
팩토리 시그니처: make(task: str, cat_cols: list[str], seed: int) -> estimator

이름 뒤에 `@<spec>` 을 붙이면 컨텍스트 선택 래퍼로 감쌈 (예: tabpfn_v2@knn1000, models/context.py 참고).
"""

from __future__ import annotations

import importlib

REGISTRY = {
    "lgbm": ("models.gbdt", "make_lgbm"),
    "xgb": ("models.gbdt", "make_xgb"),
    "catboost": ("models.gbdt", "make_catboost"),
    "tabpfn": ("models.tfm", "make_tabpfn"),  # 최신 기본 버전 (첫 실행 시 브라우저 라이선스 동의 필요)
    "tabpfn_v2": ("models.tfm", "make_tabpfn_v2"),  # 라이선스 동의 없이 사용 가능
    "tabicl": ("models.tfm", "make_tabicl"),
    "limix2": ("models.tfm", "make_limix2"),  # 400M, 비상업 라이선스
    "limix2m": ("models.tfm", "make_limix2m"),  # 2M 경량판, 비상업 라이선스
    "mitra": ("models.tfm", "make_mitra"),  # Mitra-v2 zero-shot
    "mitra_ft": ("models.tfm", "make_mitra_ft"),  # Mitra-v2 + 50 step fine-tuning
    "exaone": ("models.tfm", "make_exaone"),  # 가중치 비상업 라이선스
    "kumo": ("models.tfm", "make_kumo"),  # NVIDIA Kumo-Tabular (large)
}

_LIMIX = [["--no-deps", "git+https://github.com/limix-ldm-ai/LimiX.git"], ["einops", "kditransform", "nvtx"],
          ["-U", "numba"]]  # Kaggle 기본 numba 는 numpy<2.3 만 지원 → 다른 패키지가 numpy 를 올리면 kditransform 이 깨짐

# Kaggle 노트북에서 모델별로 추가 설치할 pip 인자 목록 (항목마다 pip install 한 번)
KAGGLE_PIP = {
    "tabpfn": [["tabpfn"]],
    "tabpfn_v2": [["tabpfn"]],
    "tabicl": [["tabicl"]],
    "limix2": _LIMIX,  # torch==2.9.1 고정을 피하려고 --no-deps 로 설치
    "limix2m": _LIMIX,
    "mitra": [["autogluon.tabular[mitra]>=1.6"]],
    "mitra_ft": [["autogluon.tabular[mitra]>=1.6"]],
    "exaone": [["exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"]],
    "kumo": [["structured-data-models"]],
}


def base_name(name: str) -> str:
    return name.partition("@")[0]


def make_model(name: str, task: str, cat_cols: list[str], seed: int):
    base, _, spec = name.partition("@")
    if base not in REGISTRY:
        raise KeyError(f"알 수 없는 모델 '{base}'. 사용 가능: {', '.join(REGISTRY)}")
    module, fn = REGISTRY[base]
    factory = getattr(importlib.import_module(module), fn)
    if not spec:
        return factory(task, cat_cols, seed)
    from models.context import wrap

    return wrap(lambda: factory(task, cat_cols, seed), spec, task, cat_cols, seed)
