# 실험 기록

Tabular Foundation Model(TFM) vs GBDT 비교. 원본 수치는 `results/results.csv`, 모델별 OOF/테스트 예측은 `results/<comp>/<model>/`.

공통 조건: 대회별 고정 fold(`folds.csv`, seed 42), 튜닝 없는 기본 설정, Kaggle Notebook GPU(Tesla T4 ×2)에서 실행.

## House Prices (1,460행, 회귀, 83 피처 중 범주형 44) — 지표: log1p(SalePrice) RMSE ↓

### 5-fold (2026-10-19)

| 모델 | RMSE | ± fold std | MAE | 시간 |
|---|---|---|---|---|
| **tabpfn_v2** | **0.1151** | 0.016 | 0.0713 | 75s |
| kumo | 0.1191 | 0.023 | 0.0703 | 66s |
| tabicl | 0.1242 | 0.021 | 0.0767 | 21s |
| catboost | 0.1250 | 0.016 | 0.0796 | 154s |
| lgbm | 0.1288 | 0.015 | 0.0842 | 21s |
| xgb | 0.1306 | 0.013 | 0.0842 | 14s |
| limix2, limix2m, mitra, mitra_ft, exaone | 실패 (아래 "알려진 문제" 참고, 수정 후 재실행 예정) | | | |

- TabPFN v2 가 5-fold 기준으로도 1위. Kumo 가 첫 실행에서 2위.
- 다만 상위 모델 간 차이가 fold std(0.013~0.023)보다 작은 경우가 많아 순위 확정은 어려움.

### 리더보드 제출 (2026-10-19, public LB, 4,288팀)

| 모델 | CV RMSE | Public LB | 순위 |
|---|---|---|---|
| **kumo** | 0.1191 | **0.10895** | 77위 (상위 1.8%) |
| tabpfn_v2 | 0.1151 | 0.11094 | 89위 (상위 2.1%) |
| catboost | 0.1250 | 0.12199 | 713위 (상위 16.6%) |

- 튜닝 없는 TFM 단일 모델이 상위 2% 안. GBDT 기본 설정보다 훨씬 위.
- CV 에서는 tabpfn_v2 > kumo 였지만 LB 에서는 뒤집힘 → 두 모델 차이는 오차 범위로 봐야 함.
- 이 대회 LB 최상위에는 0.0 점(정답 유출) 제출이 섞여 있어 실제 순위는 이보다 좋은 편.

### 1-fold 동작 확인 (참고)

tabpfn_v2 0.1177 / tabicl 0.1269 / catboost 0.1284 / lgbm 0.1359 / xgb 0.1363

## Spaceship Titanic (8,693행, 이진 분류, 17 피처) — 지표: accuracy ↑

### 5-fold

진행 중 (11개 모델). 수정 전 코드로 올라가 있어 limix2/limix2m/mitra/mitra_ft/exaone 은 실패 예상 → 별도 재실행 필요.

### 1-fold 동작 확인 (참고, fold 하나 ≈1,740행이라 accuracy 표준오차 ±0.9%p)

| 모델 | Accuracy | AUC | Logloss | 시간 |
|---|---|---|---|---|
| tabicl | 0.8252 | 0.9090 | 0.3715 | 10.5s |
| catboost | 0.8229 | 0.9110 | 0.3698 | 15.4s |
| tabpfn_v2 | 0.8212 | 0.9108 | 0.3646 | 28.1s |
| xgb | 0.8212 | 0.9080 | 0.3755 | 2.4s |
| lgbm | 0.8189 | 0.9066 | 0.3806 | 9.5s |

## 모델 목록 (`models/`)

| 이름 | 모델 | 비고 |
|---|---|---|
| lgbm / xgb / catboost | GBDT 베이스라인 | 고정 파라미터, early stopping 없음 |
| tabpfn_v2 | TabPFN v2 | v2.5 이상은 브라우저 라이선스 동의 필요 → Kaggle 에서 사용 불가 |
| tabicl | TabICL v2 | |
| limix2 / limix2m | LimiX-2 (400M) / LimiX-2M | 비상업 라이선스. `--no-deps` 설치 (torch==2.9.1 고정 회피) |
| mitra / mitra_ft | Mitra-v2 zero-shot / +50 step fine-tune | AutoGluon sklearn 인터페이스. **회귀는 v1 체크포인트** |
| exaone | EXAONE-Tabular (LG) | 가중치 비상업 라이선스 |
| kumo | NVIDIA Kumo-Tabular (large) | 회귀는 999 분위수 평균 사용 |

## 알려진 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| limix 계열 import 실패 (`Numba needs NumPy 2.2 or less`) | 다른 패키지가 numpy 를 2.5 로 올림, Kaggle 기본 numba 0.61 은 numpy<2.3 | `pip install -U numba` (0.68 은 numpy<2.6 지원) 추가 |
| mitra 회귀 `different number of output (1!=1000)` | mitra-regressor-2 는 분위수 출력, AutoGluon 1.6 sklearn 인터페이스와 불일치 | 회귀만 `autogluon/mitra-regressor`(v1) 사용 |
| exaone `No available kernel` | flash attention 고정 선택, T4(sm75)에는 해당 커널 없음 | sm80 미만 GPU 에서 SDPA 를 MATH 백엔드로 패치 |
| CatBoost `cat_features must be integer or string` | pandas 3 에서 category→str 변환 시 NaN 유지 | 결측을 `"__NA__"` 문자열로 명시 변환 |

## Kaggle 운영 메모

- 전화번호 미인증 계정은 Notebook 인터넷/GPU 가 조용히 꺼짐 (`ajagent` 계정에서 pip 설치 실패, GPU 사용량 0). 현재 `deepmachine` 계정 사용.
- 대회 데이터를 노트북에 붙이려면 해당 계정으로 Rules 동의 필요 (Spaceship, House Prices 동의 완료).
- 배치 GPU 세션은 계정당 동시 2개까지. 가끔 버전이 QUEUED 에서 진행되지 않음 → 커널 삭제 후 재 push 로 해결된 적 있음.
- `kaggle kernels push` 는 실패해도 exit code 0 → `kaggle_run.py` 에서 출력 문자열로 검사.
- Kaggle 기본 환경 (2026-10-19): Python 3.13, torch 2.11.0+cu128, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1.
- zsh 에서 모델 목록을 변수로 넘길 때는 배열(`"${M[@]}"`) 사용. 따옴표 없는 `$M` 은 단어 분리가 안 돼 인자 하나로 전달됨.

## 다음 할 일

- [ ] Spaceship 5-fold 결과 수집
- [ ] 실패한 5개 모델(limix2, limix2m, mitra, mitra_ft, exaone) 두 대회 재실행
- [x] House Prices 제출 (kumo, tabpfn_v2, catboost)
- [ ] Spaceship 제출로 리더보드 백분위 확인
- [ ] TFM + GBDT 앙상블/스태킹 효과 확인
- [ ] 패키지 버전 고정 및 실행 시 버전 기록
