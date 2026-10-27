"""Name That LLM 피처: 글 한 편(한 문단, 230~1,900자)을 표 한 행으로.

  sty_*  문체 통계: 길이, 문장·단어 길이, 구두점 빈도 (em-dash·굽은 따옴표가 모델 계열을 크게 가름)
  chr_*  글자 n-gram TF-IDF → SVD (대소문자·구두점 유지)
  wrd_*  단어 1~2-gram TF-IDF → SVD
  emb_*  고정 LM(EMB_MODEL) 은닉 상태 평균 → PCA. data/ 에 캐시

TF-IDF·SVD·PCA 는 라벨 없이 train+test 전체로 맞춤 (fold 간 라벨 누수 없음).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from km.core.competition import load_competition
from km.core.features import split

EMB_MODEL = "Qwen/Qwen3-0.6B"
N_CHR, N_WRD, N_EMB = 48, 32, 64

PUNCT = {"emdash": "—", "endash": "–", "hyphen": "-", "curly_apos": "’", "curly_quote": "[“”]", "straight_apos": "'",
         "straight_quote": '"', "semicolon": ";", "colon": ":", "comma": ",", "paren": r"\(", "excl": "!",
         "question": r"\?", "digit": r"\d", "upper": "[A-Z]"}


def stylometry(text: pd.Series) -> pd.DataFrame:
    # 숨겨진 테스트에서 처음 제출이 재실행 오류 → 빈 글·단어 없는 글에서도 깨지지 않게 (빈 배열 max, 0 나누기)
    n_chars = text.str.len().clip(lower=1)
    words = text.str.split().apply(lambda w: w or [""])
    sents = text.apply(lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t) if s])
    sent_len = sents.apply(lambda s: np.array([len(x.split()) for x in s] or [0]))
    out = pd.DataFrame({
        "sty_chars": n_chars,
        "sty_words": words.str.len(),
        "sty_sents": sents.str.len(),
        "sty_sent_mean": sent_len.apply(np.mean),
        "sty_sent_std": sent_len.apply(np.std),
        "sty_sent_max": sent_len.apply(np.max),
        "sty_word_len": words.apply(lambda w: np.mean([len(x) for x in w])),
        "sty_ttr": words.apply(lambda w: len({x.lower() for x in w}) / len(w)),
        "sty_long_words": words.apply(lambda w: np.mean([len(x) >= 10 for x in w])),
    })
    for name, pat in PUNCT.items():
        out[f"sty_{name}"] = text.str.count(pat) / n_chars * 1000  # 1,000자당
    return out


def svd(texts: pd.Series, prefix: str, n: int, **tfidf) -> pd.DataFrame:
    X = TfidfVectorizer(sublinear_tf=True, **tfidf).fit_transform(texts)
    Z = TruncatedSVD(n, random_state=0).fit_transform(X)
    return pd.DataFrame(Z, columns=[f"{prefix}_{i}" for i in range(n)], index=texts.index)


def lm_embed(texts: list[str], model_id: str, batch: int = 16) -> np.ndarray:
    """LM 마지막 층과 가운데 층의 토큰 평균을 이어 붙임 (n, 2 * hidden)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=640).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pool = [(h.float() * mask).sum(1) / mask.sum(1) for h in (hs[len(hs) // 2], hs[-1])]
            out.append(torch.cat(pool, 1).cpu().numpy())
            if (i // batch) % 50 == 0:
                print(f"    [emb] {i + len(enc['input_ids'])}/{len(texts)}", flush=True)
    return np.concatenate(out)


def cached_embed(texts: pd.Series) -> np.ndarray:
    key = hashlib.sha1(("\0".join(texts) + EMB_MODEL).encode()).hexdigest()[:12]
    path = load_competition("name-that-llm").data_dir / f"emb_{EMB_MODEL.replace('/', '_')}_{key}.npy"
    if not path.exists():
        np.save(path, lm_embed(texts.tolist(), EMB_MODEL))
    return np.load(path)


def build(train: pd.DataFrame, test: pd.DataFrame):
    # 결측은 빈 글, 짝 없는 서로게이트 등 UTF-8 로 못 바꾸는 문자는 '?' 로 (pandas 문자열·해시·토크나이저가 실패함).
    # pandas 3 문자열 열은 그런 문자를 담지 못하므로 Series 로 만들기 전에 정리
    raw = [*train["text"].tolist(), *test["text"].tolist()]
    index = pd.MultiIndex.from_tuples([("train", i) for i in range(len(train))] + [("test", i) for i in range(len(test))])
    text = pd.Series([(t if isinstance(t, str) else "").encode("utf-8", "replace").decode("utf-8") for t in raw],
                     index=index)
    E = cached_embed(text)
    emb = PCA(N_EMB, random_state=0).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-6))
    df = pd.concat([
        stylometry(text),
        svd(text, "chr", N_CHR, analyzer="char", ngram_range=(1, 4), lowercase=False, min_df=3, max_features=300_000),
        svd(text, "wrd", N_WRD, ngram_range=(1, 2), min_df=2),
        pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(N_EMB)], index=text.index),
    ], axis=1).astype(float)
    return *split(df), []
