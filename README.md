# kaggle-master

Tabular Foundation Model(TabPFN, TabICL 등)을 GBDT 베이스라인과 같은 CV fold 위에서 비교하는 Kaggle 실험 프레임워크.

## 구조

```
competitions/<slug>/config.yaml   # task, target, metric, CV 설정
competitions/<slug>/features.py   # build(train, test) -> (X, X_test, cat_cols)
competitions/<slug>/folds.csv     # 첫 실행 시 생성, 모든 모델이 공유
models/                           # 모델 레지스트리 (models/__init__.py 에 등록)
run.py                            # CV 실행 → results/<slug>/<model>/{oof,test}.npy, results/results.csv
kaggle_run.py                     # 같은 실험을 Kaggle Notebook(GPU)에서 실행 후 결과 수집
submit.py                         # 제출 파일 생성(+ --push 로 제출), 여러 모델이면 평균
download.py                       # 로컬 실행용 데이터 다운로드
```

## 준비

1. Kaggle → Settings → API → Create New Token, 토큰을 `~/.kaggle/access_token` 에 저장
2. 대회 페이지에서 Rules 동의 (예: https://www.kaggle.com/competitions/spaceship-titanic/rules)
3. Kaggle 계정 전화번호 인증 (Notebook GPU·인터넷 사용에 필요)

## 사용

```bash
# Kaggle GPU에서 실행 (빠른 확인: --max-folds 1)
uv run python kaggle_run.py spaceship-titanic lgbm xgb catboost tabpfn_v2 tabicl --max-folds 1
uv run python kaggle_run.py spaceship-titanic lgbm xgb catboost tabpfn_v2 tabicl

# 제출
uv run python submit.py spaceship-titanic catboost tabicl --push -m "cb+tabicl"
```

모델: `lgbm`, `xgb`, `catboost`, `tabpfn_v2`, `tabicl`, `tabpfn`(최신 버전, 브라우저 라이선스 동의 필요라 Kaggle에서는 사용 불가).
