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

- `@knn1000` = 전체 학습 데이터 대신 쿼리 16개 묶음마다 가장 가까운 1000행만 컨텍스트로 사용 (2026-10-21, `km/models/context.py`). 전체로는 메모리/한도 초과였던 3개 모델이 모두 GBDT 수준 이상으로 돌아감. limix2 는 넘지 못함.

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

### TFM 격차 원인 분석 (2026-10-21, fold 0, 테스트 예측 생략) — AUC

구간별로 보면 TFM 은 두 곳에서 lgbm 에 짐: (1) 양성 비율이 0·1 에 가까운 구간 (Personal Travel 0.10, Wifi=5 0.95), (2) lgbm 예측이 0.01~0.05 / 0.9~0.99 인 확신 구간(데이터의 74%). 확신 구간은 라벨 노이즈(약 3~5%)가 대부분이라 모든 모델이 0.5~0.58 이지만, 무작위 1만 TFM 은 0.52 로 거의 무작위, lgbm 은 0.57.
(확신 구간 예측을 뒤집는 실험: 상하위 1% 뒤집으면 AUC 0.958 → 0.921. 모델 확률이 이미 노이즈에 맞게 보정돼 있어 이득 없음.)

| 설정 | 전체 | Personal | Wifi=5 | p<.05 구간 | p>.9 구간 | 시간 |
|---|---|---|---|---|---|---|
| lgbm (전체 56만) | **0.9585** | 0.8327 | 0.6536 | 0.569 | 0.581 | — |
| tabpfn_v2 무작위 1만 | 0.9544 | 0.8202 | 0.6096 | 0.522 | 0.548 | — |
| tabpfn_v2 knn3000_g1024_imp | 0.9564 | 0.8164 | 0.6296 | 0.523 | 0.559 | 814s |
| tabpfn_v2 knn3000_g1024_imp_bal20 | 0.9565 | 0.8199 | 0.6278 | 0.526 | 0.556 | 896s |
| tabpfn_v2 knn6000_g1024_imp | 0.9569 | 0.8177 | 0.6411 | 0.525 | 0.565 | 1,989s |
| tabicl knn3000_g1024_imp | 0.9565 | 0.8185 | 0.6372 | 0.526 | 0.561 | 347s |
| tabicl knn3000_g1024_imp_bal20 | 0.9564 | 0.8187 | 0.6320 | 0.523 | 0.558 | 358s |
| tabicl rand10000 | 0.9553 | 0.8152 | 0.6274 | 0.524 | 0.558 | 87s |
| tabicl rand50000 | 0.9569 | 0.8240 | 0.6347 | 0.535 | 0.564 | 526s |
| tabicl rand100000 | 0.9575 | 0.8257 | 0.6384 | 0.543 | 0.568 | 1,847s |
| **tabicl rand200000** | **0.9577** | 0.8247 | 0.6476 | 0.540 | 0.571 | 7,333s |

- **bal(클래스 균형 코어셋)은 효과 없음** (±0.0001). "이웃이 한 클래스로 쏠려서 진다"는 가설은 약함.
- kNN 은 k 를 늘릴수록 이득이 줄어듦 (1000→3000 +0.0017, 3000→6000 +0.0005).
- **무작위로 많이 넣는 쪽이 kNN 보다 좋음**: tabicl 무작위 5만 = tabpfn_v2 kNN 6000 (0.9569) 인데 4배 빠름. 이 데이터에서는 "가까운 이웃"보다 "많은 데이터"가 중요.
- 무작위 컨텍스트를 키우면 kNN 이 못 고친 구간(Personal, 확신 구간 p<.05)이 함께 좋아짐 → 확신 구간 안의 미세한 순위는 전체 분포를 많이 봐야 잡힘.
- tabicl 20만 행에서 lgbm 과 차이 −0.0008 (처음 −0.004). 단 10만→20만 은 +0.0002, 시간은 4배 → 전체(56만)도 lgbm 을 넘기는 어려워 보임.

## 컨텍스트 선택 실험 (kNN 컨텍스트, 브랜치 `knn-context`)

가설: 쿼리와 비슷한 학습 샘플 k개만 컨텍스트로 넣어도 전체 데이터만큼 추론할 수 있다 → 대용량에서도 TFM 사용 가능.
구현: `km/models/context.py` (`<모델>@knn1000_g8_imp` 형식). 실행: Colab 무료 T4 (`km colab`), 테스트 예측 생략.

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

### S4E2 묶음 크기 (2026-10-21, TabICL, imp + bal20) — accuracy / logloss / 총 시간

| 설정 | 결과 |
|---|---|
| k1000 g16 | 0.9076 / 0.2774 / 1,841s |
| k1000 g64 | 0.9049 / 0.2815 / 490s |
| k1000 g256 | 0.9043 / 0.2877 / 142s |
| **k2000 g256** | **0.9078 / 0.2752 / 137s** |
| 전체 (기준) | 0.9089 / 0.2644 / 60s |

- 묶음을 키우면 조금씩 떨어지지만(g16→g256 −0.3%p), 컨텍스트를 2000행으로 늘리면 회복 이상 → g16 보다 좋고 13배 빠름.
- k2000 g256 은 전체와 0.11%p 차이, 총 시간도 전체의 2.3배 수준까지 줄어듦 → S6E10 에 쓸 설정.

### S6E10 1-fold (2026-10-22, 학습 fold 약 56만 행, 검증 약 14만 행, 이진 분류) — AUC / accuracy / logloss / 시간

전체 컨텍스트는 TabICL 에 못 넣는 규모. 기존 방식은 무작위 1만 행 컨텍스트.

| 모델 | 결과 |
|---|---|
| lgbm (전체 56만 행 학습) | **0.95849** / 0.9251 / 0.2288 / 70s |
| tabicl 무작위 1만 행 (기존) | 0.95564 / 0.9225 / 0.2365 / 79s |
| tabicl@knn2000_g256_imp_bal20 | 0.95662 / 0.9258 / 0.2312 / 819s |
| tabicl@knn4000_g256_imp_bal20 | 0.95709 / 0.9262 / 0.2299 / 1,288s |

- kNN 컨텍스트는 무작위 1만 행보다 2000행으로도 좋고(+0.10), 4000행이면 +0.15. 컨텍스트가 클수록 계속 좋아짐.
- 그래도 LightGBM 과는 AUC 0.0014 차이가 남음. accuracy 는 kNN 4000 이 lgbm 보다 높음(0.9262 vs 0.9251).
- 비용: 검증 14만 행 → 묶음 629개라 TFM fit 629회, lgbm 의 12~18배 시간.
- 첫 시도는 일반 tabicl 이 14만 행을 한 번에 예측하다 RAM OOM 으로 프로세스가 강제 종료 → `_TFM` 예측을 2만 행 단위로 나눠 해결.

## TFM 추론 속도 (2026-10-22, Databricks, fold 0, 테스트 예측 포함) — 초

TabArena 대회를 A10G(24GB)에서 돌렸더니 TFM 이 GBDT 의 10~100배 → 원인을 나눠 개선.

| 대회 | 모델 | 이전 (A10G) | 코드 개선 후 (A10G) | 피처 500개 (A10G) | H100 + 피처 500개 | 점수 (마지막 열) |
|---|---|---|---|---|---|---|
| bioresponse (1,776 피처) | tabpfn_v2 | 497 | 458 (fit 380) | 34 | **23** | logloss 0.4486 → 0.4400 |
| bioresponse | tabicl | -9 로 강제 종료 | 106 | 19 | **14** | logloss 0.3981 |
| bioresponse | kumo | — | — | — | **23** | logloss 0.3942 |
| GiveMeSomeCredit (ctx 1만, 쿼리 13만) | tabpfn_v2 | ~57 | 44 | — | **20** | AUC 0.8572 |
| GiveMeSomeCredit | tabicl | ~20 | — | — | **10** | AUC 0.8613 |
| GiveMeSomeCredit | kumo | ~250 | — | — | **36** | AUC 0.8641 |
| GiveMeSomeCredit | kumo_e4 (estimator 4개) | — | — | — | **15** | AUC 0.8641 |
| GiveMeSomeCredit | limix2 | 1,490 | 1,422 | — | 제거 | AUC 0.8639 |

- "이전" 중 ~ 는 5-fold 총 시간 / 5.
- 코드 개선: fold 당 검증·테스트를 한 번에 예측 (컨텍스트 계산 1회), TabPFN 컨텍스트 KV 캐시(GPU 에 들어갈 때만),
  Kumo 는 sdm `fit()`/`predict()` 로 컨텍스트 1회 인코딩, 모델마다 새 프로세스 (tabicl -9 해결).
  lgbm / tabpfn_v2 / tabicl 은 합쳐 예측해도 출력이 완전히 같음 (로컬 확인).
- 넓은 데이터는 피처 수가 결정적: TabPFN v2 프리셋은 피처 서브샘플링이 없어 1,776개 × 8 estimator. `tfm_max_features: 500`
  (LightGBM 중요도 상위)로 14배 빨라지고 점수도 좋아짐 (tabpfn_v2 0.4486→0.4400, tabicl 0.4276→0.3981).
- bioresponse tabpfn_v2 의 fit 380s 는 CPU 전처리. 로컬 맥은 같은 버전으로 12s → 원인 미확인, 피처 500개면 8s 라 보류.
- limix2 는 OOM 재시도를 없애도 1,422s (파이프라인 32개 × 쿼리 13만 행 연산 자체가 큼) → 실행·리더보드에서 제거.
- H100 에서는 OOM 경고 0회. kumo_e4 는 AUC 차이 0.00004 로 2.4배 빠름 (fold 하나, 대회 하나 기준).

## 의미 기반 결정 모델: LLM 객관식 · SAP-RPT (2026-10-22, 브랜치 `decision-models`)

TypeSafe Jev("System One" 결정 모델: 텍스트/JSON → 타입 있는 선택지 확률)를 오픈 모델로 흉내 낸 실험.
Jev 자체는 비공개 API 라 대신 두 방향을 시험.

- `llm*`: 행을 "열: 값" 텍스트로 바꿔 객관식 질문 → 생성 없이 forward 한 번으로 선택지 글자(A/B/…) 확률만 읽음.
  LLM 은 라벨을 안 봄(zero-shot). `llm` 은 그 확률을 로지스틱 회귀로 보정, `llm_lgbm` 은 원래 피처에 붙여 LightGBM.
  질문은 `config.yaml` 의 `llm:` 블록 (description, question, labels).
- `sap_rpt`: SAP-RPT-1-OSS (구 ConTextTab). 열 이름·셀 값을 문장 임베딩으로 쓰는 tabular ICL 모델 (라벨 컨텍스트 사용).

Titanic 은 로컬 M3 Pro, 나머지는 Databricks A10G. lgbm·기존 최고는 위 대회별 표의 값.

| 대회 (지표) | lgbm | 기존 최고 | sap_rpt | llm (4B) | llm8b | llm_lgbm | llm8b_lgbm |
|---|---|---|---|---|---|---|---|
| Titanic (acc ↑) | 0.8339 | 0.8462 exaone | **0.8384** | 0.7632 | — | 0.8272 | — |
| Spaceship (acc ↑) | 0.8101 | 0.8244 limix2 | 0.8051 | 0.7144 | 0.7208 | 0.8123 | 0.8081 |
| Spaceship (AUC ↑) | 0.9044 | 0.9191 limix2 | 0.8954 | 0.7278 | 0.7421 | 0.9043 | 0.9034 |
| S4E2 (acc ↑) | 0.9066 | 0.9117 limix2 | 0.8984 | 0.8127 | 0.8086 | 0.9028 | 0.9049 |
| S4E2 (logloss ↓) | 0.3008 | 0.2551 limix2 | 0.2991 | 0.5679 | 0.5720 | 0.3224 | 0.3194 |
| House Prices (RMSE ↓) | 0.1288 | 0.1151 tabpfn_v2 | 0.1217 | — | — | — | — |

5-fold 시간: llm 은 Titanic 1,185s(로컬), Spaceship 1,490s, S4E2 4,157s / llm8b 는 2,419s, 6,714s /
sap_rpt 는 House 49s, Spaceship 245s, S4E2 649s. llm_lgbm 은 같은 프로세스에서 llm 채점 결과를 재사용해 GBDT 시간만 듦.

- LLM zero-shot 은 상식이 통하는 데이터에서 의외로 강함: S4E2(식습관·키·몸무게 → 비만도)는 라벨 없이 7클래스 0.81.
  가상 세계인 Spaceship 은 AUC 0.73.
- 하지만 GBDT 에 피처로 붙이면 이득 없음 (모든 대회에서 차이가 fold std 안이거나 하락, logloss 는 일관되게 나빠짐).
  LLM 이 보는 정보가 이미 피처에 있는 것과 같아서 → 라벨이 수천 행 있으면 GBDT 가 같은 패턴을 직접 학습.
  가치가 있다면 피처에서 버린 텍스트 열(이름·설명), 라벨이 거의 없을 때, 학습 없이 바로 판단해야 할 때.
- 4B → 8B 는 차이 없음 (오차 범위), 시간만 1.6배.
- Qwen3-1.7B 는 내용과 무관하게 거의 항상 "B" 를 고름 (Titanic 200행 zero-shot AUC 0.55, 4B 는 0.85).
  선택지 순서를 뒤집어 함께 묻는 보정을 넣었지만 4B 이상은 순서를 바꿔도 AUC 차이 0.001 → 4B 이상이면 한 번만 물어도 됨.
- 로컬 MPS 는 4B 에서 초당 1.4~2.6행 (실효 약 4 TFLOPS, 하드웨어 한계 수준). A10G 는 약 18행/s.
  프롬프트의 51% 가 모든 행에 공통인 앞부분(시스템 + 데이터 설명)이라 prefix KV 캐시를 쓰면 더 줄일 수 있음 (미적용).
- sap_rpt 는 GBDT 기본 설정과 비슷한 수준, House Prices 에서만 GBDT 보다 확실히 좋음.
  A10G 에 맞춰 컨텍스트 2048행 × bagging 8 로 줄인 설정이라, `sap_rpt8k`(README 기본 8192행, 80GB GPU)를 H100 에서 볼 여지 있음.

## 모델 목록 (`km/models/`)

| 이름 | 모델 | 비고 |
|---|---|---|
| lgbm / xgb / catboost | GBDT 베이스라인 | 고정 파라미터, early stopping 없음 |
| tabpfn_v2 | TabPFN v2 | v2.5 이상은 브라우저 라이선스 동의 필요 → Kaggle 에서 사용 불가 |
| tabicl | TabICL v2 | |
| limix2m | LimiX-2M | 비상업 라이선스. `--no-deps` 설치 (torch==2.9.1 고정 회피). limix2(400M)는 느려서 제거 (2026-10-22) |
| mitra / mitra_ft | Mitra-v2 zero-shot / +50 step fine-tune | AutoGluon sklearn 인터페이스. **회귀는 v1 체크포인트** |
| exaone | EXAONE-Tabular (LG) | 가중치 비상업 라이선스 |
| kumo | NVIDIA Kumo-Tabular (large) | 회귀는 999 분위수 평균 사용 |
| sap_rpt / sap_rpt8k | SAP-RPT-1-OSS (구 ConTextTab) 컨텍스트 2048 / 8192 | HF gated + 연구용 라이선스. `km databricks` 가 가중치를 로컬에서 받아 Volume `weights/` 에 올림 (Kaggle 원격은 미지원) |
| llm / llm_lgbm, llm8b / llm8b_lgbm | Qwen3-4B / 8B 객관식 zero-shot (`km/models/llm.py`) | 분류 전용. 8B 는 fp16 16GB 라 T4 한 장에는 안 들어감 |

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
| sap_rpt Titanic accuracy 0.62 (다수 클래스 수준) | SAP-RPT 가 X 와 y(새 0..n 인덱스)를 인덱스 기준 concat → fold 로 자른 X 의 원래 인덱스와 라벨이 어긋남 | 넘기기 전에 `reset_index(drop=True)` |
| llm `No module named 'torch'` (Databricks) | AI Runtime 기본 환경에 torch 없음, transformers 는 torch 를 의존성으로 설치 안 함 | 설치 목록에 `torch` 추가 |
| llm8b `CUDA out of memory` (A10G 22GB) | 같은 프로세스에서 앞서 로드한 4B 가 캐시에 남은 채 8B 로드 | 새 모델을 로드할 때 이전 모델을 내림 |

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

앙상블 제출 (2026-10-21, 테스트 예측 단순 평균, 조합은 OOF 점수로 선택):

| 대회 | 조합 | OOF | Public LB | 기존 최고 | 순위 변화 |
|---|---|---|---|---|---|
| House Prices | tabpfn_v2+kumo | 0.1161 | 0.10925 | 0.10895 (kumo) | 변화 없음 |
| Spaceship | **limix2+kumo** | 0.8229 | **0.81949** | 0.81809 (limix2) | 38 → **35위** / 1,734 |
| Spaceship | limix2+kumo+catboost | 0.8237 | 0.81739 | | |
| Titanic | limix2m+limix2 | 0.8507 | 0.76794 | 0.77511 (exaone) | 변화 없음 |
| S4E2 | limix2+tabicl+kumo@knn1000 | 0.9129 | 0.91184 (private **0.91194**) | 0.91437 (private 0.91076) | 종료 대회 |
| S6E10 | **lgbm+catboost** | 0.9585 | **0.95800** | 0.95792 (lgbm) | 2,059 → **2,013위** / 2,592 |

- 앙상블 이득은 작음 (+0.0001~0.0014). OOF 상 이득이 큰 조합(Titanic +0.45%p)이 LB 에서는 오히려 떨어짐 → 작은 데이터 accuracy 조합 선택은 과적합.
- S4E2 는 public 은 낮지만 private 은 단일 limix2 보다 높음 → 최종 기준(private)에서는 앙상블이 더 좋음.

- House Prices 는 이미 제출함 (위 표: kumo 0.10895, tabpfn_v2 0.11094).
- Titanic 은 CV 와 LB 차이가 7%p — 418행 테스트라 변동이 크고, CV 1위 차이도 오차 범위였음.

## Kaggle 운영 메모

- 전화번호 미인증 계정은 Notebook 인터넷/GPU 가 조용히 꺼짐 (`ajagent` 계정에서 pip 설치 실패, GPU 사용량 0). 현재 `deepmachine` 계정 사용.
- 대회 데이터를 노트북에 붙이려면 해당 계정으로 Rules 동의 필요 (Spaceship, House Prices 동의 완료).
- 배치 GPU 세션은 계정당 동시 2개까지. 가끔 버전이 QUEUED 에서 진행되지 않음 → 커널 삭제 후 재 push 로 해결된 적 있음.
- `kaggle kernels push` 는 실패해도 exit code 0 → `km kaggle` (`km/remote/kaggle.py`) 에서 출력 문자열로 검사.
- Kaggle 기본 환경 (2026-10-19): Python 3.13, torch 2.11.0+cu128, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1.
- Kaggle 세션에는 T4 가 2장 → `km run` 이 GPU 수를 감지해 GPU 마다 작업자를 두고 모델을 나눠 동시 실행 (`--gpus N` 으로 조절). 큐에 대기 중인 버전도 동시 실행 2개 한도에 포함됨.
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
- [x] S6E10 TFM 격차 원인 분석 (bal, k 크기, 무작위 컨텍스트 크기 — fold 0)
- [ ] ~~S6E10 kNN 5-fold + 제출~~ → 무작위 대용량 컨텍스트가 더 좋아 tabicl rand 쪽으로 대체
- [ ] S6E10 tabicl 대용량 컨텍스트 5-fold + 제출 (rand100000 이면 fold 당 약 30분 + 테스트 30만 행 예측)
- [x] House Prices 제출 (kumo, tabpfn_v2, catboost)
- [x] 나머지 4개 대회 CV 1위 모델 제출
- [ ] Spaceship 제출로 리더보드 백분위 확인
- [ ] TFM + GBDT 앙상블/스태킹 효과 확인
- [ ] 패키지 버전 고정 및 실행 시 버전 기록
- [ ] sap_rpt8k (컨텍스트 8192) H100 에서 5-fold — 컨텍스트를 키우면 TFM 상위권에 붙는지
