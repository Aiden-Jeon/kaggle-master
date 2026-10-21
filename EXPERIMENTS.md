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
| mitra_ft | 0.1263 | 0.028 | 0.0759 | 7,995s |
| exaone | 0.1275 | 0.032 | 0.0721 | 204s |
| lgbm | 0.1288 | 0.015 | 0.0842 | 21s |
| mitra | 0.1289 | 0.031 | 0.0764 | 259s |
| limix2 | 0.1300 | 0.039 | 0.0707 | 328s |
| xgb | 0.1306 | 0.013 | 0.0842 | 14s |
| limix2m | 0.1311 | 0.034 | 0.0741 | 58s |

- TabPFN v2 가 5-fold 기준으로도 1위. Kumo 가 첫 실행에서 2위.
- 재실행 모델(2026-10-21) 결과: RMSE 는 중위권이지만 MAE 는 limix2(0.0707)·exaone(0.0721)이 상위권 → 대부분은 잘 맞히고 일부 이상치 집에서 크게 틀림 (fold std 0.03~0.04 로 큼).
- mitra_ft 는 fine-tuning 으로 mitra 대비 RMSE 0.003 개선, 대신 시간은 30배 (2시간+).
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

### 5-fold (2026-10-19)

| 모델 | Accuracy | ± fold std | AUC | Logloss | 시간 |
|---|---|---|---|---|---|
| **limix2** | **0.8245** | 0.005 | **0.9192** | 0.3509 | 3,503s |
| kumo | 0.8237 | 0.007 | **0.9192** | **0.3488** | 273s |
| tabpfn_v2 | 0.8135 | 0.007 | 0.9056 | 0.3764 | 92s |
| mitra_ft | 0.8130 | 0.002 | 0.9030 | 0.3819 | 7,587s |
| limix2m | 0.8128 | 0.009 | 0.9039 | 0.3810 | 80s |
| catboost | 0.8125 | 0.008 | 0.9048 | 0.3825 | 66s |
| tabicl | 0.8123 | 0.011 | 0.9038 | 0.3815 | 58s |
| lgbm | 0.8102 | 0.006 | 0.9044 | 0.3848 | 16s |
| xgb | 0.8096 | 0.008 | 0.9045 | 0.3824 | 9s |
| mitra | 0.8059 | 0.006 | 0.8959 | 0.3988 | 265s |
| exaone@knn1000 | 0.8151 | 0.009 | 0.9080 | 0.3721 | 7,486s |

- limix2(400M) 와 Kumo 가 공동 1위권 (AUC 동일 0.9192). 단 limix2 는 Kumo 보다 13배 느림.
- Kumo 가 accuracy +1%p, AUC +0.014 로 나머지와 확실히 벌어짐 (fold std 0.007 보다 큼).
- Kumo 를 빼면 TFM 과 GBDT 는 0.81 근처에서 오차 범위 안.

### 1-fold 동작 확인 (참고, fold 하나 ≈1,740행이라 accuracy 표준오차 ±0.9%p)

| 모델 | Accuracy | AUC | Logloss | 시간 |
|---|---|---|---|---|
| tabicl | 0.8252 | 0.9090 | 0.3715 | 10.5s |
| catboost | 0.8229 | 0.9110 | 0.3698 | 15.4s |
| tabpfn_v2 | 0.8212 | 0.9108 | 0.3646 | 28.1s |
| xgb | 0.8212 | 0.9080 | 0.3755 | 2.4s |
| lgbm | 0.8189 | 0.9066 | 0.3806 | 9.5s |

## Titanic (891행, 이진 분류, 12 피처) — 지표: accuracy ↑

### 5-fold (2026-10-20)

| 모델 | Accuracy | ± fold std | AUC | 시간 |
|---|---|---|---|---|
| **exaone** | **0.8462** | 0.015 | 0.8934 | 30s |
| tabicl | 0.8440 | 0.014 | 0.8878 | 9s |
| catboost | 0.8418 | 0.019 | 0.8749 | 21s |
| limix2 | 0.8395 | 0.018 | **0.9017** | 257s |
| xgb | 0.8384 | 0.016 | 0.8839 | 3s |
| tabpfn_v2 | 0.8350 | 0.018 | 0.8755 | 14s |
| lgbm | 0.8339 | 0.016 | 0.8821 | 10s |
| kumo | 0.8328 | 0.018 | 0.8888 | 24s |
| limix2m | 0.8440 | 0.016 | 0.8839 | 19s |
| mitra_ft | 0.8339 | 0.019 | 0.8825 | 709s |
| mitra | 0.8294 | 0.023 | 0.8830 | 59s |

- 891행이라 fold std(±1.5~1.9%p)가 모델 간 차이(최대 1.3%p)보다 큼 → accuracy 순위는 의미 없음.
- AUC 로 보면 LimiX-2 가 가장 높음 (0.902).

## Playground S4E2 (20,758행, 7-class 분류, 17 피처) — 지표: accuracy ↑

### 5-fold (2026-10-20, T4 ×2 병렬)

| 모델 | Accuracy | ± fold std | Logloss | 시간 |
|---|---|---|---|---|
| **limix2** | **0.9118** | 0.004 | **0.2551** | 478분 |
| tabicl | 0.9090 | 0.005 | 0.2644 | 2분 |
| kumo@knn1000 | 0.9088 | 0.005 | 0.2677 | 181분 |
| exaone@knn1000 | 0.9087 | 0.004 | 0.2687 | 182분 |
| xgb | 0.9077 | 0.003 | 0.2686 | 7분 |
| tabpfn_v2@knn1000 | 0.9072 | 0.006 | 0.2905 | 93분 |
| lgbm | 0.9067 | 0.003 | 0.3009 | 4분 |
| catboost | 0.9042 | 0.005 | 0.2782 | 16분 |
| mitra | 0.8980 | 0.003 | 0.3021 | 23분 |
| limix2m | 0.8971 | 0.006 | 0.3050 | 5분 |
| mitra_ft | 0.8969 | 0.007 | 0.2973 | 185분 |

- `@knn1000` = 전체 학습 데이터 대신 쿼리 16개 묶음마다 가장 가까운 1000행만 컨텍스트로 사용 (2026-10-21, `models/context.py`). 전체로는 메모리/한도 초과였던 3개 모델이 모두 GBDT 수준 이상으로 돌아감. limix2 는 넘지 못함.

- LimiX-2 가 1위지만 8시간 소요 (TabICL 은 2분에 2위). 비용 대비로는 TabICL 이 압도적.
- 상위 4개(limix2, tabicl, xgb, lgbm) 차이는 fold std 수준.
- 실패 원인: tabpfn_v2 는 1만 행 공식 한도 초과로 거부, exaone 은 math 어텐션이 445GiB 요청, kumo 는 13.7GiB 요청으로 T4 메모리 초과.

## Playground S6E10 (699,635행, 이진 분류, 22 피처, 진행 중 대회) — 지표: AUC ↑ (추정)

TFM 은 fold 마다 학습 데이터에서 1만 행만 무작위로 뽑아 컨텍스트로 사용 (`tfm_max_rows: 10000`). GBDT 는 전체(fold 당 ~56만 행) 사용.

### 5-fold (2026-10-20, T4 ×2 병렬)

| 모델 | AUC | ± fold std | Accuracy | Logloss | 시간 |
|---|---|---|---|---|---|
| **lgbm** | **0.95838** | 0.0006 | 0.9256 | 0.2286 | 9분 |
| catboost | 0.95820 | 0.0007 | 0.9255 | 0.2292 | 33분 |
| xgb | 0.95815 | 0.0007 | 0.9253 | 0.2295 | 14분 |
| limix2m | 0.95442 | 0.0009 | 0.9209 | 0.2411 | 40분 |
| tabpfn_v2 | 0.95411 | 0.0005 | 0.9208 | 0.2422 | 100분 |
| mitra | 0.95195 | 0.0008 | 0.9173 | 0.2486 | 244분 |
| tabicl, kumo | 메모리 부족으로 실패 → 예측 분할 코드는 준비됨, 재실행은 **보류** | | | | |

### kNN 컨텍스트 탐색 (2026-10-21, fold 0 만, 테스트 예측 생략) — AUC

| 설정 | AUC | 시간 |
|---|---|---|
| lgbm (전체 56만 행, 기준) | **0.9585** | — |
| tabpfn_v2@knn3000_g1024_imp | 0.9564 | 814s |
| tabicl@knn1000_g256_imp | 0.9559 | 886s |
| tabpfn_v2@knn1000_g256_imp | 0.9556 | 812s |
| tabpfn_v2@knn1000_g1024_imp | 0.9547 | 305s |
| tabpfn_v2 무작위 1만 행 (기존) | 0.9544 | — |

- kNN 은 모든 설정에서 무작위 1만 행보다 좋음 (컨텍스트 3분의 1 로 +0.002). 그래도 전체 데이터 GBDT 와 0.002 차이.
- 묶음 g256 → g1024 는 −0.001 정도, k 를 늘리는 쪽이 더 효과적.
- 5-fold + 테스트 예측이면 모델당 약 3.5시간 (fold 당 검증 14만 + 테스트 30만 행).

- 데이터가 크면 GBDT 가 확실히 우위 (AUC +0.004, fold std 의 5배 이상). 컨텍스트를 전체의 1.8% 로 줄인 TFM 은 데이터를 다 못 씀.
- TFM 은 예측 행이 많아 느림 (fold 당 검증 14만 + 테스트 30만 행). Mitra 는 4시간.

## 컨텍스트 선택 실험 (kNN 컨텍스트, 브랜치 `knn-context`)

가설: 쿼리와 비슷한 학습 샘플 k개만 컨텍스트로 넣어도 전체 데이터만큼 추론할 수 있다 → 대용량에서도 TFM 사용 가능.
구현: `models/context.py` (`<모델>@knn1000_g8_imp` 형식). 실행: Colab 무료 T4 (`colab_run.py`), 테스트 예측 생략.

### Spaceship 5-fold (2026-10-20, 학습 fold 약 6,950행) — accuracy / AUC / logloss

| k | TabPFN 무작위 k | TabPFN kNN k (g8, imp) |
|---|---|---|
| 256 | 0.7664 / 0.8436 / 0.4906 | 0.8014 / 0.8940 / 0.3989 |
| 512 | 0.7881 / 0.8668 / 0.4528 | 0.8087 / 0.9005 / 0.3858 |
| 1000 | 0.7916 / 0.8807 / 0.4303 | 0.8117 / 0.9046 / 0.3779 |
| 2000 | 0.7983 / 0.8919 / 0.4050 | 0.8107 / 0.9064 / 0.3740 |
| 전체 | 0.8136 / 0.9056 / (미기록) | |

TabICL: 전체 0.8115 / 0.9032 / 0.3836, 무작위 1000 0.7956 / 0.8815 / 0.4291, kNN 1000 은 fold 0~3 만 완료(0.8229 / 0.8022 / 0.8148 / 0.8147, 평균 0.8137 — fold 4 실행 중 런타임 끊김, 전체 5-fold 평균과 fold 가 달라 직접 비교는 주의).
fold std 는 accuracy 기준 0.005~0.017.

- 모든 k 에서 kNN > 무작위 (accuracy +1.2~3.5%p, logloss 크게 개선). kNN 256 이 무작위 2000 보다 좋음.
- kNN 1000 은 전체와 accuracy 0.2%p 차이(fold std 0.75%p 이내), kNN 2000 은 AUC 가 전체보다 높음.
- 비용: kNN 은 쿼리 8개 묶음마다 재-fit → fold 당 3~7.5분 (전체는 수십 초). 이점은 전체가 컨텍스트에 안 들어가는 대용량에서만 의미 있음.
- 1-fold 스모크(fold 0): 거리 공간 imp(LightGBM 중요도 가중)가 raw 보다 좋음 (TabPFN 0.8212 vs 0.8194).

### S4E2 5-fold (2026-10-21, 학습 fold 약 16,600행, 7클래스) — accuracy / logloss

| 컨텍스트 | TabICL | TabPFN v2 |
|---|---|---|
| 전체 | 0.9089 / 0.2644 (60s) | 불가 (1만 행 한도 초과) |
| 무작위 1000 | 0.8872 / 0.3369 | 0.8818 / 0.3493 |
| kNN 1000 (g16, imp) | 0.9064 / 0.2750 (1,799s) | 0.9055 / 0.2918 (1,303s) |

같은 fold 의 GBDT (원래 폴더 결과): xgb 0.9077, lgbm 0.9067, catboost 0.9042. fold std 0.004~0.006.

- 컨텍스트 6%(1000/16,600)로 무작위 대비 +1.9~2.4%p, TabICL 은 전체와 −0.25%p (fold std 이내).
- TabPFN v2 는 전체 데이터를 못 넣는 크기인데, kNN 1000 으로 GBDT 수준(0.9055)에 도달 → 가설의 실용적 의미가 처음 드러남.
- 다수 클래스 비율 0.68 (7클래스인데도 이웃이 한 클래스로 크게 쏠림). logloss 는 전체보다 약간 나쁨.

다음: S6E10(70만 행)에서 전체 데이터 GBDT 와 비교. kNN 비용(fold 당 4~6분, 쿼리 수에 비례) 줄이는 방법 필요.

### S4E2 검색 공간·구성 방식 비교 (2026-10-21, TabICL, 컨텍스트 1000행) — accuracy / logloss / 총 시간

| 방식 | 결과 |
|---|---|
| 전체 (기준) | 0.9089 / 0.2644 / 60s |
| kNN imp (기존) | 0.9064 / 0.2750 / 1,799s |
| **kNN leaf** | 0.9073 / 0.2753 / 2,034s |
| **kNN imp + bal20** | **0.9076** / 0.2774 / 1,841s (다수 클래스 비율 0.68 → 0.59) |
| part imp (영역 17개) | 0.8945 / 0.3182 / 139s |
| part leaf | 0.8816 / 0.3556 / 130s |
| 무작위 (기준) | 0.8872 / 0.3369 / 14s |

TabPFN v2 part leaf: 0.8752 / 0.4040 (kNN imp 0.9055 대비 크게 낮음).

- leaf 공간·bal 코어셋은 kNN imp 대비 +0.1%p 수준 (fold std 0.4%p 이내, 유의하지 않음). 전체와 차이는 0.13%p 까지 줄어듦.
- bal 은 이웃 쏠림을 줄였지만 logloss 는 개선되지 않음.
- **영역 분할(part)은 실패**: 13배 빠르지만 무작위와 비슷하거나 더 나쁨. 영역 중심 하나에 가까운 1000행은 영역 가장자리 쿼리에 맞지 않음 → "쿼리마다 가까움"이 핵심.
- 결론: 비용을 줄이려면 쿼리 단위 지역성을 유지하는 방향(묶음 크기 g 를 키우는 정도)으로 가야 함.

다음: S4E2 에서 g16 → g64 → g256 로 성능이 얼마나 떨어지는지 측정 → S6E10 적용 가능성 판단.

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
| limix2m `KeyError: 'tf_mlp_layer_type'` | `stable-ai/LimiX-2M` 체크포인트가 현재 LimiX 코드와 설정 키 불일치 | README 가 가리키는 `stable-ai/LimiX-1_2M` 사용 |
| mitra `NotFittedError` (Pipeline) | sklearn 1.9 Pipeline 의 fitted 검사를 Mitra 가 통과 못 함 | Pipeline 대신 `_Imputed` 래퍼로 직접 결측 처리 |
| kumo `CUDA out of memory (83 GiB)`, tabicl 작업자 무응답 종료 (S6E10) | 44만 행을 한 번에 예측 | TFM 예측을 2만 행씩 나눠 처리 (`PREDICT_CHUNK`), LimiX/Kumo 는 모델을 한 번만 로드 |
| tabpfn_v2 `Number of samples 16,606 ... greater than 10,000` (S4E2) | TabPFN v2 공식 컨텍스트 한도 1만 행 | `ignore_pretraining_limits=True` |
| exaone `CUDA out of memory (445 GiB)` (S4E2) | T4 대체용 math 어텐션이 어텐션 행렬을 통째로 생성 | mem-efficient 커널(sm75 지원) 우선, 안 되는 head 폭만 math |
| exaone `CUDA out of memory (157 GiB)` (Spaceship) | mem-efficient 패치 후에도 발생, math 대체 경로로 빠지는 것으로 추정 | 미해결, 보류 |
| kumo `CUDA out of memory (13.7 GiB)` (S4E2) | 컨텍스트 1.6만 행 + 질의 2만 행 | Kumo 만 예측 청크 2,048행 |
| 결과 수집 작업이 멈춤 | 맥 잠자기 중 kaggle CLI 호출이 무한 대기 | `kaggle()` 호출에 10분 timeout |
| CatBoost `cat_features must be integer or string` | pandas 3 에서 category→str 변환 시 NaN 유지 | 결측을 `"__NA__"` 문자열로 명시 변환 |

## 대회별 CV 1위 모델 제출 (2026-10-21)

| 대회 | 모델 | 지표 | CV | Public LB | Private LB | 순위 (public) |
|---|---|---|---|---|---|---|
| Spaceship Titanic | limix2 | accuracy | 0.8245 | 0.81809 | — | 38 / 1,722 (상위 2.2%) |
| Titanic | exaone | accuracy | 0.8462 | 0.77511 | — | 6,994 / 11,375 (동점자 많음, 점수 기준 4,587위) |
| S4E2 (비만, 종료) | limix2 | accuracy | 0.9118 | 0.91437 | 0.91076 | 종료 후 제출 → 순위 미반영, public 점수 기준 916 / 3,589 (상위 25.5%) |
| S6E10 (항공 만족도, 진행 중) | lgbm | AUC | 0.9584 | 0.95792 | — | 2,028 / 2,544 (상위 80%) |

kNN 컨텍스트 모델 추가 제출 (2026-10-21):

| 대회 | 모델 | CV | Public LB | Private LB |
|---|---|---|---|---|
| S4E2 | kumo@knn1000 | 0.9088 | 0.91076 | 0.90959 |
| S4E2 | exaone@knn1000 | 0.9087 | 0.91076 | 0.90688 |
| S4E2 | tabpfn_v2@knn1000 | 0.9072 | 0.90787 | 0.90634 |
| Spaceship | exaone@knn1000 | 0.8151 | 0.81014 (약 105위) | — |

- 모두 기존 제출(limix2)보다 낮음. CV 순서와 LB 순서가 대체로 일치.

- House Prices 는 이미 제출함 (위 표: kumo 0.10895, tabpfn_v2 0.11094).
- Titanic 은 CV 와 LB 차이가 7%p — 418행 테스트라 변동이 크고, CV 1위 차이도 오차 범위였음.

## Kaggle 운영 메모

- 전화번호 미인증 계정은 Notebook 인터넷/GPU 가 조용히 꺼짐 (`ajagent` 계정에서 pip 설치 실패, GPU 사용량 0). 현재 `deepmachine` 계정 사용.
- 대회 데이터를 노트북에 붙이려면 해당 계정으로 Rules 동의 필요 (Spaceship, House Prices 동의 완료).
- 배치 GPU 세션은 계정당 동시 2개까지. 가끔 버전이 QUEUED 에서 진행되지 않음 → 커널 삭제 후 재 push 로 해결된 적 있음.
- `kaggle kernels push` 는 실패해도 exit code 0 → `kaggle_run.py` 에서 출력 문자열로 검사.
- Kaggle 기본 환경 (2026-10-19): Python 3.13, torch 2.11.0+cu128, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1.
- Kaggle 세션에는 T4 가 2장 → `run.py` 가 GPU 수를 감지해 GPU 마다 작업자를 두고 모델을 나눠 동시 실행 (`--gpus N` 으로 조절). 큐에 대기 중인 버전도 동시 실행 2개 한도에 포함됨.
- `machine_shape: NvidiaTeslaP100` 으로 올렸지만 실제로는 T4 ×2 에서 실행됨 (S6E10 로그 기준) → GPU 종류 지정이 반영되지 않는 것으로 보임.
- zsh 에서 모델 목록을 변수로 넘길 때는 배열(`"${M[@]}"`) 사용. 따옴표 없는 `$M` 은 단어 분리가 안 돼 인자 하나로 전달됨.

## 다음 할 일

- [x] Spaceship 5-fold 결과 수집
- [x] S6E10 5-fold (tabicl, kumo 제외)
- [x] S4E2 5-fold (tabpfn_v2, exaone, kumo 제외)
- [x] Titanic 실패 모델 재실행 (limix2m, mitra, mitra_ft)
- [x] Spaceship·House Prices 실패 모델 재실행 (Spaceship exaone 은 메모리 부족)
- [x] S4E2 tabpfn_v2·exaone·kumo, Spaceship exaone → kNN 컨텍스트로 해결
- [x] S6E10 kNN 컨텍스트 탐색 (1-fold)
- [ ] S6E10 kNN 5-fold + 제출 (모델당 3.5시간, GBDT 를 넘을 가능성 낮음 — 앙상블용이면 실행)
- [x] House Prices 제출 (kumo, tabpfn_v2, catboost)
- [x] 나머지 4개 대회 CV 1위 모델 제출
- [ ] Spaceship 제출로 리더보드 백분위 확인
- [ ] TFM + GBDT 앙상블/스태킹 효과 확인
- [ ] 패키지 버전 고정 및 실행 시 버전 기록
