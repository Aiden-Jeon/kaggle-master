# kaggle-master

Tabular Foundation Model(TabPFN, TabICL 등)을 GBDT 베이스라인과 같은 CV fold 위에서 비교하는 Kaggle 실험 프레임워크.

## 구조

```
competitions/<slug>/config.yaml   # task, target, metric, CV 설정
competitions/<slug>/features.py   # build(train, test) -> (X, X_test, cat_cols)
competitions/<slug>/folds.csv     # 첫 실행 시 생성, 모든 모델이 공유
km/
  cli.py                          # `km <명령>` 진입점
  run.py                          # km run: CV 실행 → results/<slug>/<model>/{oof,test}.npy, results/results.csv
  submit.py                       # km submit: 제출 파일 생성(+ --push 로 제출), 여러 모델이면 평균
  leaderboard.py                  # km lb: Kaggle 제출 기록을 모아 리더보드 순위 계산
  download.py                     # km download: 로컬 실행용 데이터 다운로드
  competition.py, features.py, metrics.py, results.py   # 대회 설정·데이터·fold, 공통 피처, 지표, 결과 저장
  models/                         # 모델 레지스트리 (km/models/__init__.py 에 등록), @spec 컨텍스트 선택 (context.py)
  remote/                         # km kaggle / colab / databricks: 같은 실험을 원격 GPU 에서 실행 후 결과 수집
```

## 준비

1. Kaggle → Settings → API → Create New Token, 토큰을 `~/.kaggle/access_token` 에 저장
2. 대회 페이지에서 Rules 동의 (예: https://www.kaggle.com/competitions/spaceship-titanic/rules)
3. Kaggle 계정 전화번호 인증 (Notebook GPU·인터넷 사용에 필요)

## 사용

명령 목록은 `uv run km`, 명령별 인자는 `uv run km <명령> -h`.

```bash
# 로컬 실행
uv run km download spaceship-titanic
uv run km run spaceship-titanic lgbm catboost --max-folds 1

# Kaggle GPU에서 실행 (빠른 확인: --max-folds 1)
uv run km kaggle spaceship-titanic lgbm xgb catboost tabpfn_v2 tabicl --max-folds 1
uv run km kaggle spaceship-titanic lgbm xgb catboost tabpfn_v2 tabicl

# Colab 무료 T4: 노트북 생성 → 실행 후 셀 출력을 저장해 결과 수집
uv run km colab make spaceship-titanic tabicl --max-folds 1 -o run.ipynb
uv run km colab collect output.txt

# Databricks AI Runtime(서버리스 GPU)에서 실행 — databricks.local.yaml 에 profile, volume 지정 (git 제외)
uv run km databricks spaceship-titanic lgbm catboost tabpfn_v2 --accelerator GPU_1xA10

# 제출
uv run km submit spaceship-titanic catboost tabicl --push -m "cb+tabicl"

# 제출 기록 → 리더보드 순위 (results/leaderboard.csv), --submit N: 미제출 모델을 CV 순으로 N개 제출 후 집계
uv run km lb
uv run km lb bioresponse --submit 2
```

모델: `lgbm`, `xgb`, `catboost`, `tabpfn_v2`, `tabicl`, `tabpfn`(최신 버전, 브라우저 라이선스 동의 필요라 Kaggle에서는 사용 불가).
