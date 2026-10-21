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
