# kaggle-master

Tabular Foundation Model(TabPFN, TabICL, Kumo 등)을 GBDT 베이스라인(LightGBM, XGBoost, CatBoost)과
같은 CV fold 위에서 비교하는 Kaggle 실험 프레임워크.

- 대회마다 `config.yaml` + `features.py` 만 쓰면 모든 모델이 같은 fold·같은 피처로 돌아감
- 튜닝 없는 기본 설정으로 비교 (모델 자체의 기본 성능을 보려는 목적)
- 같은 명령으로 로컬, Kaggle Notebook, Colab, Databricks AI Runtime(서버리스 GPU)에서 실행하고 결과를 한곳(`results/`)에 모음
- 저장된 테스트 예측으로 제출하고, 제출 기록에서 리더보드 순위를 집계

## 결과 요약

13개 대회, 모두 5-fold CV. 대회별 표·분석은 [EXPERIMENTS.md](EXPERIMENTS.md),
정렬·필터·차트로 보는 리더보드는 https://aiden-jeon.github.io/kaggle-master/ (`docs/leaderboard.html`, GitHub Pages).
`(ctx 전체)` · `(ctx 20만)` = TFM 컨텍스트 상한(`tfm_max_rows: 10000`)을 풀고 다시 돌린 kumo.

| 대회 | 학습 행 | 지표 | CV 1위 | 최고 LB 순위 |
|---|---|---|---|---|
| Restaurant Revenue | 137 | RMSE ↓ | limix2m | 상위 13.9% (limix2m) |
| Titanic | 891 | acc ↑ | exaone | 상위 17.4% (kumo) |
| House Prices | 1,460 | RMSE ↓ | tabpfn_v2 | **상위 1.9%** (kumo) |
| Bioresponse | 3,751 | logloss ↓ | causilo | **1위** (kumo·tabicl·causilo) |
| Mercedes-Benz | 4,209 | R² ↑ | exaone | **3위 · 상위 0.1%** (kumo) |
| Spaceship Titanic | 8,693 | acc ↑ | kumo | 상위 2.2% (kumo) |
| S4E2 | 20,758 | acc ↑ | causilo | 상위 7.2% (kumo@knn1000) |
| Amazon | 32,769 | AUC ↑ | catboost | 상위 4.3% (catboost) |
| Otto | 61,878 | logloss ↓ | kumo (ctx 전체) | **1위** (kumo (ctx 전체)) |
| GiveMeSomeCredit | 150,000 | AUC ↑ | kumo (ctx 전체) | **1위** (kumo (ctx 전체)) |
| Allstate | 188,318 | MAE ↓ | kumo (ctx 전체) | **1위** (kumo (ctx 전체)) |
| S6E1 | 630,000 | RMSE ↓ | kumo (ctx 20만) | 상위 17.0% (kumo (ctx 20만)) |
| S6E10 | 699,635 | AUC ↑ | kumo (ctx 20만) | 상위 65.6% (kumo (ctx 20만)) |

LB 순위는 `km lb` 집계: 종료된 대회는 private 점수, 진행 중·상시 대회 (Titanic·House Prices·Spaceship·S6E10) 는 public 점수 기준.
종료된 대회의 순위는 늦은 제출 점수로 매긴 위치 (Otto·GiveMeSomeCredit·Allstate·Bioresponse 1위, Mercedes 3위 포함).

- 작은 데이터에서는 TFM 이 GBDT 와 같거나 앞섬. 큰 데이터의 GBDT 우위는 대부분 TFM 컨텍스트 1만 행 상한 탓 — 상한을 푼 kumo 가 Otto·GiveMeSomeCredit·Allstate·S6E1 에서 GBDT 최고를 넘고 S6E10 은 동률.
- TFM 중에서는 kumo 가 가장 꾸준함. 2026-10-27 에 비어 있던 조합(exaone·causilo·limix2m·mitra·mitra_ft 등 45개)을 채우자 CV 1위가 4개 대회에서 바뀜 (Restaurant limix2m, Bioresponse·S4E2 causilo, Mercedes exaone) — 모두 2위와의 차이가 fold std 안. 범주 ID 만 있는 데이터(Amazon)는 상한을 풀어도 catboost 가 앞섬 (원인은 추정).
- 상위 모델 간 차이는 대부분 fold std 안 → 작은 대회일수록 CV 순위가 LB 에서 자주 뒤집힘.

## 구조

```
competitions/<slug>/config.yaml   # task, target, metric, CV 설정
competitions/<slug>/features.py   # build(train, test) -> (X, X_test, cat_cols)
competitions/<slug>/folds.csv     # 첫 실행 시 생성, 모든 모델이 공유
km/
  cli.py                          # `km <명령>` 진입점
  core/                           # 기반 라이브러리 (명령은 km download 하나)
    competition.py, features.py, metrics.py, results.py   # 대회 설정·데이터·fold, 공통 피처, 지표, 결과 저장
    download.py                   # km download: 로컬 실행용 데이터 다운로드 (원격 런처도 사용)
    migrate.py                    # km migrate: 일회성 데이터 마이그레이션 (test-info: test.json 없는 예측에 출처 채움)
  models/                         # 모델 레지스트리 (km/models/__init__.py 에 등록), @spec 컨텍스트 선택 (context.py)
  runner/                         # 실험 실행
    run.py                        # km run: CV 실행 → results/<slug>/<model>/{oof,test}.npy, results/results.csv
    kaggle.py, colab.py, databricks.py   # km kaggle / colab / databricks: 같은 실험을 원격 GPU 에서 실행 후 결과 수집
    common.py                     # run 과 원격 런처가 공유하는 인자·코드 묶음
  leaderboard/                    # 제출과 순위
    submit.py                     # km submit: 제출 파일 생성(+ --push 로 제출), 여러 모델이면 평균
    rank.py                       # km lb: Kaggle 제출 기록을 모아 리더보드 순위 계산
    board.py, board.html          # km board: 모델별 성능·실행 시간 리더보드 (docs/leaderboard.md, 정렬·차트가 있는 docs/leaderboard.html)
docker/air/                       # Databricks 실행 이미지 (패키지 미리 설치)
results/
  results.csv                     # 실행마다 한 줄 (CV 점수, fold std, 시간, 실행 위치, 테스트 예측 여부). timestamp 는 UTC
  leaderboard.csv                 # 제출별 public/private 점수와 순위, 제출을 만든 실행(run = results.csv timestamp)
  <slug>/<model>/                 # OOF·테스트 예측, test.json = test.npy 를 만든 실행 (km submit 이 제출 설명에 적음, git 제외)
                                  # git 제외 파일 (<slug>/, _air/runs.json) 은 worktree 에서 실행·수집해도 메인 체크아웃의 results/ 에 씀
                                  # (worktree 를 지워도 남음, KM_OUTPUTS 로 위치 지정). results.csv·leaderboard.csv 는 체크아웃마다 (PR 로 합침)
EXPERIMENTS.md                    # 실험 결과와 분석
docs/known-issues.md              # 겪은 오류와 해결, Kaggle·Databricks 운영 메모
```

## 준비

Python 환경은 uv 로 관리 (`uv sync`). 로컬 의존성(`pyproject.toml`)은 GBDT·TabPFN·TabICL 까지이고,
나머지 TFM 은 원격 실행 때 설치됨 (`KAGGLE_PIP`, `docker/air/requirements.txt`).

1. Kaggle → Settings → API → Create New Token, 토큰을 `~/.kaggle/access_token` 에 저장
2. 대회 페이지에서 Rules 동의 (예: https://www.kaggle.com/competitions/spaceship-titanic/rules)
3. Kaggle 계정 전화번호 인증 (Notebook GPU·인터넷 사용에 필요)
4. (Databricks 사용 시) databricks CLI 로그인 후 저장소 루트에 `databricks.local.yaml` 작성 (git 제외)

```yaml
profile: <~/.databrickscfg 프로필>
volume: /Volumes/<catalog>/<schema>/<volume>   # 대회 데이터와 실행 결과를 둘 UC Volume
accelerator: GPU_1xH100                         # 선택. 기본 GPU 종류 (--accelerator 가 우선)
image: <catalog>.<schema>.km-air:v4             # 선택. 패키지를 미리 설치한 이미지. 없으면 실행마다 pip 설치
```

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

# Databricks AI Runtime(서버리스 GPU)에서 실행
# GPU 종류는 --accelerator 또는 databricks.local.yaml 의 accelerator (기본 GPU_1xH100).
# GPU 1장이어도 모델마다 새 프로세스로 실행, GPU 여러 장이면 모델을 GPU별로 나눠 동시 실행
# GBDT(lgbm·xgb·catboost)는 GPU 작업에 넣지 않고 CPU serverless 작업으로 따로 제출 (GPU 노드에선 CPU 경합으로 느림)
uv run km databricks spaceship-titanic lgbm catboost tabpfn_v2
# 오래 걸리는 5-fold: GPU 모델을 fold 마다 작업 하나로 동시에 제출 → 다 끝나면 fold 를 합쳐 results.csv 에 한 줄
uv run km databricks allstate-claims-severity exaone mitra_ft --split-folds

# 제출 (여러 모델이면 테스트 예측 평균)
uv run km submit spaceship-titanic catboost tabicl --push -m "cb+tabicl"

# 제출 기록 → 리더보드 순위 (results/leaderboard.csv), --submit N: 미제출 모델을 CV 순으로 N개 제출 후 집계
# 아직 안 낸 마지막 실행의 test.npy 가 없으면 (results.csv 는 test=True) 대회마다 목록으로 알림
uv run km lb
uv run km lb bioresponse --submit 2

# test.json 이 없는 예전 예측에 출처(CV·run) 채움 (1회. 이미 있는 것은 건너뜀, 확인 못 하면 비움)
uv run km migrate test-info
```

공통 실행 인자: `--max-folds N` (앞 N개 fold 만), `--no-test` (테스트 예측 생략), `--note` (results.csv 에 남길 메모),
`--fold k` (fold 하나만 → `results/<대회>/<모델>/folds/<k>/`, 다 모이면 `km run <대회> <모델> --merge-folds` 로 합침.
`km databricks --split-folds`·`--fetch`, `km kaggle --fold` 는 마지막 fold 를 받을 때 자동으로 합침).

## 모델

`km/models/__init__.py` 의 `REGISTRY` 에 등록된 이름으로 지정.

| 이름 | 모델 | 비고 |
|---|---|---|
| `lgbm` / `xgb` / `catboost` | GBDT 베이스라인 | 고정 파라미터, early stopping 없음 |
| `tabpfn_v2` | TabPFN v2 | 라이선스 동의 없이 사용 가능 |
| `tabpfn` / `tabpfn35` | TabPFN 최신 / 3.5 | 브라우저 라이선스 동의 필요 → Kaggle 에서 사용 불가 |
| `tabicl` | TabICL v2 | |
| `kumo` | NVIDIA Kumo-Tabular (large) | 회귀는 999 분위수 평균 (지표가 MAE 면 중앙값). `kumo_e4` 는 estimator 4개 (2.4배 빠름) |
| `exaone` | EXAONE-Tabular (LG) | 가중치 비상업 라이선스 |
| `limix2m` | LimiX-2M | 비상업 라이선스. `--no-deps` 설치 (torch==2.9.1 고정 회피) |
| `mitra` / `mitra_ft` | Mitra-v2 zero-shot / +50 step fine-tune | AutoGluon sklearn 인터페이스. 회귀는 v1 체크포인트 |
| `causilo` | Nums AI Causilo | 가중치 비상업 라이선스 |
| `sap_rpt` / `sap_rpt8k` | SAP-RPT-1-OSS (구 ConTextTab), 컨텍스트 2048 / 8192 | HF gated + 연구용 라이선스. `km databricks` 가 가중치를 로컬에서 받아 Volume `weights/` 에 올림 (Kaggle 원격은 미지원) |
| `llm` / `llm_lgbm`, `llm8b` / `llm8b_lgbm` | Qwen3-4B / 8B 객관식 zero-shot (`km/models/llm.py`) | 분류 전용. `_lgbm` 은 선택지 확률을 피처로 붙여 LightGBM. 질문은 config 의 `llm:` 블록 |

**컨텍스트 선택**: TFM 이름 뒤에 `@<spec>` 을 붙이면 학습 데이터 일부만 컨텍스트로 넘김
(예: `tabicl@knn2000_g256_imp_bal20`, `tabicl@rand100000`). 형식은 `km/models/context.py` 참고.

**새 모델 추가**: `km/models/__init__.py` 의 `REGISTRY` 에 `이름: (모듈, 팩토리 함수명)` 등록, Kaggle 원격 실행에
추가 설치가 필요하면 `KAGGLE_PIP` 에도 추가. 같은 모델의 설정 변형은 `VARIANTS` 에.

## 대회 추가

`competitions/<kaggle-slug>/` 에 두 파일을 만듦.

`config.yaml`

```yaml
name: titanic
task: binary              # binary | multiclass | regression
target: Survived
id_col: PassengerId       # 테스트 파일 id 열 (null 이면 1 부터 행 번호)
metric: accuracy          # 리더보드 지표 (km/core/metrics.py)
cv:
  n_splits: 5
  seed: 42
  stratify: true
# 선택
# target_transform: log1p       # 회귀 타깃 변환 (학습·평가는 변환 스케일, 제출 시 역변환)
# submit_proba: true            # 분류에서 레이블 대신 확률 제출
# submit_columns: [Id, Prob]    # 제출 파일 열 이름 (기본 [id_col, target])
# train_file / test_file        # 기본 train.csv / test.csv
# tfm_max_rows: 10000           # TFM 컨텍스트 행 상한 (큰 데이터, 무작위 샘플)
# tfm_max_features: 500         # TFM 피처 상한 (넓은 데이터, LightGBM 중요도 상위)
# llm: {description, question, labels}   # llm* 모델 질문
```

`features.py`: `build(train, test) -> (X, X_test, cat_cols)`. `km.core.features` 의 `auto_types`(문자열 → category)와
`split`(concat 한 train/test 다시 나누기)을 쓰면 짧게 끝남 (예: `competitions/titanic/features.py`).

## Databricks 실행 이미지

실행마다 pip 설치를 하지 않도록 패키지를 미리 넣은 이미지(`docker/air/`)를 Unity Catalog 에 올려 두고 씀.
모델 가중치는 Volume 의 `cache/models.tar` 에 모여 이미지와 상관없이 다시 받지 않음.

```bash
# 빌드 (linux/amd64. pypi.org 가 막힌 네트워크면 uv 설정을 secret 으로 넘김 → 이미지에는 남지 않음)
# attestation 을 끄지 않으면 OCI index 가 되어 업로드 마지막(manifest)에서 500
docker build --platform linux/amd64 --provenance=false --sbom=false \
  --secret id=uv,src=$HOME/.config/uv/uv.toml -t km-air:v4 docker/air
# Unity Catalog 에 업로드 → databricks.local.yaml 에 image: <catalog>.<schema>.km-air:v4
databricks air images push --source km-air:v4 --catalog <catalog> --schema <schema> --artifact km-air:v4
# 이미지 없이(기본 환경 + pip 설치) 실행하려면
uv run km databricks titanic lgbm --image none
```

모델 패키지를 추가·변경하면 `docker/air/requirements.txt` 를 고치고 태그를 올려 다시 빌드.
Artifact Registry 는 워크스페이스 Preview 가 켜져 있어야 함 (e2-demo-field-eng 는 업로드가 403 → FEVM 워크스페이스 사용).
