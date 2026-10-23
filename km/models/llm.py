"""LLM 결정 모델 (TypeSafe Jev 같은 'System One' 방식 흉내).

행을 "열: 값" 텍스트로 바꿔 객관식 질문(A/B/…)을 던지고, 생성 없이 한 번의 forward 로 다음 토큰의 선택지 글자
확률만 읽음. LLM 은 라벨을 보지 않으므로(zero-shot) 학습 fold 는 보정·결합에만 사용 → fold 간 누수 없음.

  llm       선택지 log 확률 → 로지스틱 회귀로 보정
  llm_lgbm  원래 피처 + 선택지 log 확률 → LightGBM  (llm8b* 와 함께 models/__init__.py 의 VARIANTS)

채점 결과는 모델별로 디스크(KM_LLM_CACHE, 기본 ~/.cache/km/llm)에 저장 → GPU 에서 모델마다 따로 뜨는 `km run`
프로세스 사이(llm → llm_lgbm)와 fold 사이에서 다시 채점하지 않음.

질문은 competitions/<slug>/config.yaml 의 llm 블록:
  llm:
    description: 데이터 설명 (선택)
    question: 레코드마다 던질 질문 (없으면 타깃 이름으로 질문)
    labels: {클래스 값: 선택지 문구}  (없으면 클래스 값 그대로)
분류 대회 전용 (회귀는 지원 안 함).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

QWEN3_4B = "Qwen/Qwen3-4B"  # 1.7B 는 내용과 무관하게 한 글자만 고름 (Titanic zero-shot AUC 0.55)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = "You are a fast, calibrated decision model. Answer with the letter of the single most likely option."

CACHE = Path(os.environ.get("KM_LLM_CACHE", Path.home() / ".cache" / "km" / "llm"))

_LOADED: dict[str, tuple] = {}
_SCORES: dict[str, dict[str, np.ndarray]] = {}  # 모델 → {프롬프트 해시: 선택지 log 확률}


def _key(text: str) -> str:
    return hashlib.sha1(f"{SYSTEM}\n{text}".encode()).hexdigest()  # 시스템 프롬프트가 바뀌면 다른 키


def _cache_path(model_id: str) -> Path:
    return CACHE / f"{model_id.replace('/', '--')}.pkl"


def _cache(model_id: str) -> dict[str, np.ndarray]:
    if model_id not in _SCORES:
        path = _cache_path(model_id)
        _SCORES[model_id] = pickle.loads(path.read_bytes()) if path.exists() else {}
    return _SCORES[model_id]


def _save(model_id: str):
    """디스크 내용과 합쳐 원자적으로 저장 (동시에 뜬 다른 프로세스가 쓴 항목을 덮어쓰지 않도록)."""
    path = _cache_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**(pickle.loads(path.read_bytes()) if path.exists() else {}), **_SCORES[model_id]}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(merged))
    tmp.replace(path)


def _fmt(v) -> str:
    if pd.isna(v):
        return "unknown"
    return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)  # float32 도 짧게


def row_texts(X: pd.DataFrame, prompt: dict) -> list[str]:
    head = f"{prompt['description'].strip()}\n\n" if prompt["description"] else ""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(prompt["options"]))
    cols = list(X.columns)
    return [f"{head}Record:\n" + "\n".join(f"- {c}: {_fmt(v)}" for c, v in zip(cols, row))
            + f"\n\nQuestion: {prompt['question']}\n{opts}" for row in X.itertuples(index=False)]


def _load(model_id: str):
    if model_id not in _LOADED:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _LOADED.clear()  # 한 번에 한 모델만 메모리에 (4B 를 둔 채 8B 를 올리면 A10 22GB 를 넘음)
        if torch.cuda.is_available():  # T4 등 sm80 미만은 bf16 이 느림 → fp16
            torch.cuda.empty_cache()
            dev, dtype = "cuda", torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
        elif torch.backends.mps.is_available():
            dev, dtype = "mps", torch.float16
        else:
            dev, dtype = "cpu", torch.float32
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")  # 왼쪽 패딩 → 마지막 위치가 답 자리
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
        _LOADED[model_id] = tok, model, dev
    return _LOADED[model_id]


def score(model_id: str, texts: list[str], n_options: int, batch: int = 16) -> np.ndarray:
    """텍스트마다 선택지 글자의 log 확률 (n, n_options). 선택지 밖 토큰은 버리고 선택지끼리 다시 정규화."""
    cache = _cache(model_id)
    keys = [_key(t) for t in texts]
    todo = list({k: t for k, t in zip(keys, texts) if k not in cache}.items())  # 중복 제거한 (키, 텍스트)
    if todo:
        import torch

        tok, model, dev = _load(model_id)
        ids = [tok.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(n_options)]
        chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for _, t in todo]
        order = np.argsort([len(c) for c in chats])  # 길이가 비슷한 것끼리 묶어 패딩 최소화
        t0 = time.time()
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                logits = model(**enc, logits_to_keep=1).logits[:, -1, ids].float()
            for i, row in zip(idx, torch.log_softmax(logits, -1).cpu().numpy()):
                cache[todo[i][0]] = row
        _save(model_id)
        print(f"    [llm] {model_id}: {len(todo)}행 채점 ({time.time() - t0:.0f}s, "
              f"{len(todo) / max(time.time() - t0, 1e-9):.1f}행/s)", flush=True)
    return np.stack([cache[k] for k in keys])


class _LLM(BaseEstimator):
    def __init__(self, task, cat_cols, seed, prompt, model_id, head):
        self.task, self.cat_cols, self.seed = task, cat_cols, seed
        self.prompt, self.model_id, self.head = prompt, model_id, head

    def _feats(self, X):
        """선택지를 정방향·역방향 순서로 두 번 물음 (작은 LLM 은 글자 위치 편향이 커서 한 순서로는 거의 상수).
        (n, 2k): 앞 k 열 정방향, 뒤 k 열 역방향 결과를 원래 클래스 순서로 되돌린 것."""
        opts = self.prompt["options"]
        fwd = score(self.model_id, row_texts(X, self.prompt), len(opts))
        rev = score(self.model_id, row_texts(X, {**self.prompt, "options": opts[::-1]}), len(opts))[:, ::-1]
        return np.hstack([fwd, rev])

    def _join(self, X, F):
        return X.assign(**{f"llm_{i}": F[:, i] for i in range(F.shape[1])})

    def fit(self, X, y):
        if self.task == "regression" or not self.prompt or not self.prompt["options"]:
            raise ValueError("LLM 결정 모델은 분류 대회만 지원 (prompt 의 options 필요)")
        if len(self.prompt["options"]) > len(LETTERS):
            raise ValueError(f"클래스 {len(self.prompt['options'])}개: 선택지 글자(A~Z) 26개를 넘는 다중 분류는 지원 안 함")
        F = self._feats(X)
        k = len(self.prompt["options"])
        zero_shot = (F[:, :k] + F[:, k:]).argmax(1)  # 두 순서 평균
        print(f"    [llm] zero-shot 정확도 {np.mean(zero_shot == np.asarray(y)):.4f} (학습 fold, 보정 전)", flush=True)
        if self.head == "lgbm":
            from km.models.gbdt import make_lgbm

            self.model_ = make_lgbm(self.task, self.cat_cols, self.seed).fit(self._join(X, F), y)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model_ = LogisticRegression(max_iter=1000).fit(F, y)
        return self

    def _in(self, X):
        F = self._feats(X)
        return self._join(X, F) if self.head == "lgbm" else F

    def predict(self, X):
        return self.model_.predict(self._in(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._in(X))


def make_llm(task, cat_cols, seed, prompt, model_id=QWEN3_4B, head="lr"):
    return _LLM(task, cat_cols, seed, prompt, model_id, head)
