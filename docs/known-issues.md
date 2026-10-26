# 알려진 문제와 운영 메모

실험 결과는 [EXPERIMENTS.md](../EXPERIMENTS.md), 사용법은 [README.md](../README.md).

## 알려진 문제와 해결

### 모델

| 문제 | 원인 | 해결 |
|---|---|---|
| limix 계열 import 실패 (`Numba needs NumPy 2.2 or less`) | 다른 패키지가 numpy 를 2.5 로 올림, Kaggle 기본 numba 0.61 은 numpy<2.3 | `pip install -U numba` (0.68 은 numpy<2.6 지원) 추가 |
| limix2m `KeyError: 'tf_mlp_layer_type'` | `stable-ai/LimiX-2M` 체크포인트가 현재 LimiX 코드와 설정 키 불일치 | README 가 가리키는 `stable-ai/LimiX-1_2M` 사용 |
| mitra 회귀 `different number of output (1!=1000)` | mitra-regressor-2 는 분위수 출력, AutoGluon 1.6 sklearn 인터페이스와 불일치 | 회귀만 `autogluon/mitra-regressor`(v1) 사용 |
| mitra `NotFittedError` (Pipeline) | sklearn 1.9 Pipeline 의 fitted 검사를 Mitra 가 통과 못 함 | Pipeline 대신 `_Imputed` 래퍼로 직접 결측 처리 |
| exaone `No available kernel` | flash attention 고정 선택, T4(sm75)에는 해당 커널 없음 | sm80 미만 GPU 에서 SDPA 를 MATH 백엔드로 패치 |
| exaone `CUDA out of memory (445 GiB)` (S4E2) | T4 대체용 math 어텐션이 어텐션 행렬을 통째로 생성 | mem-efficient 커널(sm75 지원) 우선, 안 되는 head 폭만 math |
| exaone `CUDA out of memory (157 GiB)` (Spaceship) | mem-efficient 패치 후에도 발생, math 대체 경로로 빠지는 것으로 추정 | 미해결, 보류 (kNN 컨텍스트 `@knn1000` 으로 우회) |
| kumo `CUDA out of memory (83 GiB)`, tabicl 작업자 무응답 종료 (S6E10) | 44만 행을 한 번에 예측 | TFM 예측을 2만 행씩 나눠 처리 (`PREDICT_CHUNK`), LimiX/Kumo 는 모델을 한 번만 로드 |
| kumo `CUDA out of memory (13.7 GiB)` (S4E2) | 컨텍스트 1.6만 행 + 질의 2만 행 | Kumo 만 예측 청크 2,048행 |
| tabpfn_v2 `Number of samples 16,606 ... greater than 10,000` (S4E2) | TabPFN v2 공식 컨텍스트 한도 1만 행 | `ignore_pretraining_limits=True` |
| CatBoost `cat_features must be integer or string` | pandas 3 에서 category→str 변환 시 NaN 유지 | 결측을 `"__NA__"` 문자열로 명시 변환 |
| sap_rpt Titanic accuracy 0.62 (다수 클래스 수준) | SAP-RPT 가 X 와 y(새 0..n 인덱스)를 인덱스 기준 concat → fold 로 자른 X 의 원래 인덱스와 라벨이 어긋남 | 넘기기 전에 `reset_index(drop=True)` |
| llm8b `CUDA out of memory` (A10G 22GB) | 같은 프로세스에서 앞서 로드한 4B 가 캐시에 남은 채 8B 로드 | 새 모델을 로드할 때 이전 모델을 내림 |

### 실행 · 데이터

| 문제 | 원인 | 해결 |
|---|---|---|
| 결과 수집 작업이 멈춤 | 맥 잠자기 중 kaggle CLI 호출이 무한 대기 | `kaggle()` 호출에 10분 timeout |
| Allstate `Expected 132 fields in line 188321, saw 133` | 데이터 묶음 안에 `train.csv` 와 깨진 옛 버전 `train.csv.zip` 이 같이 있어 zip 이 정상 파일을 덮어씀 | `km/core/download.py` 의 `extract()` 를 `km download`·Kaggle·Colab 이 같이 씀: zip 안의 zip 까지 풀되, 이미 있는 파일과 같은 zip 안에 압축 안 된 쌍이 있는 x.zip 은 건너뜀 |
| `km lb --submit` 가 이미 낸 모델을 또 제출, 순위 집계 0행 | kaggle CLI 가 JSON 앞에 버전 경고 줄을 출력 → `startswith("[")` 검사 실패로 제출 기록이 늘 빈 목록 | `[` 로 시작하는 줄부터 JSON 배열을 찾고, "No submissions found" 외에 못 읽으면 예외 (빈 목록으로 보고 다시 제출하지 않음). 이 버그로 신규 5개 대회에 21건이 중복 제출됨 (점수는 같음) |
| llm `No module named 'torch'` (Databricks) | AI Runtime 기본 환경에 torch 없음, transformers 는 torch 를 의존성으로 설치 안 함 | 설치 목록에 `torch` 추가 |
| GBDT 가 Databricks GPU 노드에서 극도로 느림 (Restaurant 110행 xgb fold 하나 593s, 로컬 0s) | 모델별 프로세스 6개가 한 노드에서 동시 실행, lgbm·xgb 가 각자 전체 코어만큼 스레드를 띄워 CPU 경합 | GBDT 는 CPU serverless 작업으로 따로 제출 (`km databricks` 가 자동 분리), GPU 는 기본 H100 1장. 병렬 작업자마다 CPU 스레드를 CPU 수 / 작업자 수로 제한 (`OMP_NUM_THREADS`·`KM_THREADS`, Kaggle·로컬에도 적용) |
| serverless CPU 작업이 정상 종료인데 FAILED | IPython 위에서 실행돼 `sys.exit(0)` 도 예외로 처리 | 종료 코드가 0 이 아닐 때만 `sys.exit` |

## 운영 메모

### Kaggle

- 전화번호 미인증 계정은 Notebook 인터넷/GPU 가 조용히 꺼짐 (`ajagent` 계정에서 pip 설치 실패, GPU 사용량 0). 현재 `deepmachine` 계정 사용.
- 대회 데이터를 노트북에 붙이려면 해당 계정으로 Rules 동의 필요 (Spaceship, House Prices 동의 완료).
- 배치 GPU 세션은 계정당 동시 2개까지. 가끔 버전이 QUEUED 에서 진행되지 않음 → 커널 삭제 후 재 push 로 해결된 적 있음.
- `kaggle kernels push` 는 실패해도 exit code 0 → `km kaggle` (`km/runner/kaggle.py`) 에서 출력 문자열로 검사.
- Kaggle 기본 환경 (2026-10-19): Python 3.13, torch 2.11.0+cu128, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1.
- Kaggle 세션에는 T4 가 2장 → `km run` 이 GPU 수를 감지해 GPU 마다 작업자를 두고 모델을 나눠 동시 실행 (`--gpus N` 으로 조절). 큐에 대기 중인 버전도 동시 실행 2개 한도에 포함됨.
- `machine_shape: NvidiaTeslaP100` 으로 올렸지만 실제로는 T4 ×2 에서 실행됨 (S6E10 로그 기준) → GPU 종류 지정이 반영되지 않는 것으로 보임.
- Kaggle API 는 private LB 순위를 주지 않음 → `km lb` 의 순위는 public LB 기준.

### Databricks

- e2-demo-field-eng 는 Artifact Registry 업로드가 403 (관리자 권한 없음, Preview 미활성으로 추정) → FEVM 워크스페이스로 이동.
- FEVM 에서는 Previews 의 "AI Runtime Beta Features" 를 켜야 UC 이미지 실행 가능 (켠 뒤 반영까지 수 분).
- 베이스 이미지(dcs-base-aws-runtime)에 없어 추가한 것: libgomp1 (lightgbm), gcc·libc6-dev·python3.12-dev (kumo 의 triton 컴파일).
- 이미지 빌드는 attestation 을 꺼야 함 (`--provenance=false --sbom=false`). 켜면 OCI index 가 되어 업로드 마지막(manifest)에서 500.

### 셸

- zsh 에서 모델 목록을 변수로 넘길 때는 배열(`"${M[@]}"`) 사용. 따옴표 없는 `$M` 은 단어 분리가 안 돼 인자 하나로 전달됨.
