"""대회 설정, 데이터, CV fold, 피처 모듈 로딩."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
COMP_DIR = ROOT / "competitions"


@dataclass
class Competition:
    name: str
    task: str
    target: str
    id_col: str | None  # 테스트 파일의 id 열. 없으면(null) 1 부터 매긴 행 번호
    metric: str
    cv: dict
    target_transform: str | None = None  # 회귀 타깃 변환 (예: log1p)
    submit_proba: bool = False  # 분류 대회에서 레이블 대신 양성 클래스 확률 제출
    tfm_max_rows: int | None = None  # TFM 컨텍스트 행 상한 (큰 데이터용), GBDT 에는 적용 안 함
    tfm_max_features: int | None = None  # TFM 피처 상한 (넓은 데이터용), 넘으면 LightGBM 중요도 상위만 사용
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    submit_columns: list[str] | None = None  # 제출 파일 열 이름 [id, 예측]. 기본은 [id_col, target]

    @property
    def dir(self) -> Path:
        return COMP_DIR / self.name

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary", "multiclass")


TRANSFORMS = {"log1p": (np.log1p, np.expm1)}  # name: (forward, inverse)


def load_competition(name: str) -> Competition:
    cfg = yaml.safe_load((COMP_DIR / name / "config.yaml").read_text())
    return Competition(**cfg)


def load_data(comp: Competition) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (comp.data_dir / comp.train_file).exists():
        raise FileNotFoundError(
            f"{comp.data_dir}/{comp.train_file} 없음. 먼저 `uv run km download {comp.name}` 실행"
        )
    return pd.read_csv(comp.data_dir / comp.train_file), pd.read_csv(comp.data_dir / comp.test_file)


def load_features(comp: Competition) -> ModuleType:
    """competitions/<name>/features.py 를 모듈로 로드. build(train, test) 를 제공해야 함."""
    spec = importlib.util.spec_from_file_location(f"features_{comp.name}", comp.dir / "features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_folds(comp: Competition, y: np.ndarray) -> np.ndarray:
    """fold 번호 배열. 모든 모델이 같은 fold를 쓰도록 folds.csv 로 고정 저장."""
    path = comp.dir / "folds.csv"
    if path.exists():
        folds = pd.read_csv(path)["fold"].to_numpy()
        if len(folds) == len(y):
            return folds
        raise ValueError(f"{path} 의 길이({len(folds)})가 train({len(y)})과 다름")

    cv = comp.cv
    splitter_cls = StratifiedKFold if cv.get("stratify") and comp.is_classification else KFold
    splitter = splitter_cls(n_splits=cv["n_splits"], shuffle=True, random_state=cv["seed"])
    folds = np.empty(len(y), dtype=int)
    for i, (_, va) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[va] = i
    pd.DataFrame({"fold": folds}).to_csv(path, index=False)
    return folds
