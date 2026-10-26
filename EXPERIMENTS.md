# 실험 기록

Tabular Foundation Model(TFM) vs GBDT 비교. 원본 수치는 `results/results.csv`, 리더보드 제출 기록은 `results/leaderboard.csv`,
모델별 OOF/테스트 예측은 `results/<comp>/<model>/`. 모델 이름은 [README 의 모델 목록](README.md#모델) 참고.
시간·s/1K·LB 를 한 표로 모은 리더보드는 [docs/leaderboard.md](docs/leaderboard.md) (`km board` 로 생성).

**공통 조건**: 대회별 고정 fold(`folds.csv`, seed 42), 튜닝 없는 기본 설정. 대회별 결과는 5-fold,
주제별 실험 중 일부(S6E10 컨텍스트 탐색, 추론 속도, 실행 준비 시간, Bioresponse)는 fold 0 만 — 표마다 표시.

**실행 환경** (`results.csv` 의 `where` 열. 환경이 다르면 시간 열은 직접 비교 불가):

| 기간 | 환경 | 대상 |
|---|---|---|
| 2026-10-19 ~ 22 | Kaggle Notebook (T4 ×2), Colab 무료 T4 | House Prices, Spaceship, Titanic, S4E2, S6E10, 컨텍스트 선택 |
| 2026-10-22 | Databricks AI Runtime A10G → H100 | GiveMeSomeCredit, Amazon, Bioresponse, 결정 모델, 추론 속도 |
| 2026-10-23 ~ | Databricks: TFM 은 H100 1장, GBDT 는 CPU serverless 4코어 | Restaurant, Mercedes, Otto, Allstate, S6E1 |
| 2026-10-26 | Databricks H100 (kumo 만): 컨텍스트 상한 해제·20만 행 재실행 | Amazon, Otto, GiveMeSomeCredit, Allstate, S6E1, S6E10 |

2026-10-23 이후 대회는 **같은 표 안에서도** GBDT(CPU 4코어)와 TFM(H100) 시간을 직접 비교하기 어려움.
2026-10-26 kumo 재실행 행은 같은 H100 의 1만 행 kumo 행과 비교 가능 (Amazon·GiveMeSomeCredit 의 1만 행 kumo 는 A10G,
S6E10 은 1만 행 kumo 가 없고 나머지 행이 Kaggle T4 라 비교 불가).

**표기**: 굵게 = 1위, `± fold std` = fold 간 표준편차, 시간 = 5-fold 합계 (테스트 예측 포함).
`kumo (ctx 전체)` · `kumo (ctx 20만)` = `tfm_max_rows` 상한을 푼 kumo 재실행 (`results.csv` 의 `@full` · `@rand200000`). 상한이 걸린 나머지 TFM 은 1만 행 무작위.
LB 순위는 표마다 집계 시점을 적음 — `km lb` 집계(`results/leaderboard.csv`)와 제출 당시 기록은 팀 수가 달라 순위가 조금씩 다름.

실행 중 겪은 오류와 해결, Kaggle·Databricks 운영 메모는 [docs/known-issues.md](docs/known-issues.md).

## 목차

- [요약](#요약)
- [대회별 결과](#대회별-결과) (학습 행 수 순):
  [Restaurant Revenue](#restaurant-revenue) · [Titanic](#titanic) · [House Prices](#house-prices) ·
  [Bioresponse](#bioresponse) · [Mercedes-Benz](#mercedes-benz) · [Spaceship Titanic](#spaceship-titanic) ·
  [S4E2](#playground-s4e2) · [Amazon](#amazon-employee-access) · [Otto](#otto) · [GiveMeSomeCredit](#givemesomecredit) ·
  [Allstate](#allstate-claims-severity) · [S6E1](#playground-s6e1) · [S6E10](#playground-s6e10)
- [주제별 실험](#주제별-실험):
  [컨텍스트 선택](#컨텍스트-선택) · [외부 벤치마크 대조](#외부-벤치마크-대조) · [의미 기반 결정 모델](#의미-기반-결정-모델) ·
  [TFM 추론 속도](#tfm-추론-속도) · [Databricks 실행 준비 시간](#databricks-실행-준비-시간)
- [다음 할 일](#다음-할-일)

## 요약

| 대회 | 학습 행 | 과제 · 지표 | CV 1위 | GBDT 최고 | TFM 최고 | 최고 LB 순위 (public) |
|---|---|---|---|---|---|---|
| Restaurant Revenue | 137 | 회귀 · RMSE ↓ | kumo 2,426,650 | catboost 2,640,319 | kumo | tabpfn_v2 345 / 2,260 (15.3%) |
| Titanic | 891 | 이진 · acc ↑ | exaone 0.8462 | catboost 0.8418 | exaone | exaone 4,623 / 11,490 (40.2%) |
| House Prices | 1,460 | 회귀 · RMSE ↓ | tabpfn_v2 0.1151 | catboost 0.1250 | tabpfn_v2 | kumo 79 / 4,372 (1.8%) |
| Bioresponse | 3,751 | 이진 · logloss ↓ | 5-fold 미실행 | 미실행 | (fold 0) kumo 0.3942 | 미제출 |
| Mercedes-Benz | 4,209 | 회귀 · R² ↑ | kumo 0.5742 | catboost 0.5564 | kumo | kumo 1,801 / 3,825 (47.1%) |
| Spaceship Titanic | 8,693 | 이진 · acc ↑ | kumo 0.8237 | catboost 0.8125 | kumo | exaone@knn1000 103 / 1,746 (5.9%) |
| S4E2 | 20,758 | 7-class · acc ↑ | tabicl 0.9090 | xgb 0.9077 | tabicl | kumo@knn1000 · exaone@knn1000 1,452 / 3,590 (40.4%) |
| Amazon | 32,769 | 이진 · AUC ↑ | catboost 0.8957 | catboost | kumo 0.8795 (ctx 전체) | catboost 64 / 1,689 (3.8%) |
| Otto | 61,878 | 9-class · logloss ↓ | kumo 0.3774 (ctx 전체) | lgbm 0.4648 | kumo 0.3774 (ctx 전체) | kumo (ctx 전체) 1 / 3,510 (<0.1%) |
| GiveMeSomeCredit | 150,000 | 이진 · AUC ↑ | kumo 0.8694 (ctx 전체) | catboost 0.8660 | kumo 0.8694 (ctx 전체) | kumo (ctx 전체) 1 / 926 (0.1%) |
| Allstate | 188,318 | 회귀 · MAE ↓ | kumo 1,114.2 (ctx 전체) | lgbm 1,188.3 | kumo 1,114.2 (ctx 전체) | kumo (ctx 전체, 중앙값) 1 / 3,048 (<0.1%) |
| S6E1 | 630,000 | 회귀 · RMSE ↓ | kumo 8.6767 (ctx 20만) | xgb 8.7736 | kumo 8.6767 (ctx 20만) | kumo (ctx 20만) 764 / 4,320 (17.7%) |
| S6E10 | 699,635 | 이진 · AUC ↑ | kumo 0.95880 (ctx 20만) | lgbm 0.95838 | kumo 0.95880 (ctx 20만) | kumo (ctx 20만) 2,009 / 3,071 (65.4%) |

순위는 `results/leaderboard.csv` 기준. 진행 중 대회는 집계 시점에 따라 바뀌고,
종료된 대회는 늦은 제출이라 순위표에는 안 들어가므로 점수로 매긴 위치. `(ctx …)` 는 상한을 푼 kumo 재실행 (2026-10-26).

**지금까지의 결론**

- **큰 데이터의 GBDT 우위는 대부분 컨텍스트 상한 탓 (2026-10-26 재실행)**: 원인은 6개 대회 config 의 `tfm_max_rows: 10000`.
  TabArena 의 Kumo 래퍼 기본 상한은 20만 행. 작은 데이터(Restaurant, Titanic, House, Mercedes, Spaceship)에서는 TFM 이 GBDT 와 같거나 위.
  큰 데이터는 TFM 이 1만 행만 보고 있었음. 상한을 풀자 kumo 가 Otto (logloss 0.3774 vs lgbm 0.4648), GiveMeSomeCredit (AUC 0.8694 vs catboost 0.8660),
  Allstate (MAE 1,114.2 vs lgbm 1,188.3), S6E1 (RMSE 8.6767 vs xgb 8.7736) 에서 GBDT 최고를 넘음.
  S6E10 은 20만 행 (fold 학습의 약 36%) 에서 lgbm 과 0.0004 차이 — fold std 이내 동률.
- **Amazon 은 상한을 풀어도 격차가 남음**: kumo 0.8274 → 0.8795 (fold 학습 2.6만 행 전부) 로 올랐지만 catboost 0.8957 과 AUC 0.016 차이 (fold std 의 약 2배).
  9개 열이 모두 범주 ID 라 범주 처리 방식 차이가 원인일 것으로 추정 — 검증 안 함. 외부 BeyondArena 고카디널리티 부분집합에서도 CatBoost 가 강함.
- **상한 해제의 비용**: kumo 5-fold 시간이 2~5.3배 (Otto 150→302s, Allstate 212→959s, S6E1 369→1,952s, 모두 H100).
  tabicl·tabpfn_v2 는 아직 1만 행 상한 그대로라 TFM 간 순위는 같은 조건이 아님 (재실행 진행 중, [다음 할 일](#다음-할-일)).
- **TFM 중에서는 kumo 가 가장 꾸준함**: 2026-10-23 배치 5개 대회에서 모두 CV 기준 TFM 1위. 전체 1위는 재실행 전 3개 (Restaurant·Mercedes·Allstate) → 재실행 후 5개 (Otto·S6E1 추가).
- **상위 모델 간 차이는 대부분 fold std 안**: 작은 대회일수록 CV 순위가 LB 에서 자주 뒤집힘 (House, Restaurant, Titanic).
- **컨텍스트 선택**: kNN 컨텍스트는 무작위보다 확실히 좋고 TFM 을 한도 밖 크기(S4E2)에서도 쓰게 해 주지만,
  70만 행(S6E10)에서는 무작위로 많이 넣는 쪽이 kNN 보다 낫고, kumo 20만 행 5-fold 는 GBDT 와 동률 (위 항목).
- **사전학습 데이터 중복 가능성**: Otto·GiveMeSomeCredit·Amazon 같은 공개 데이터는 TFM 사전학습에 들어갔을 수 있어 CV 가 부풀었을 수 있음 (확인 안 함).
  테스트 라벨은 공개된 적이 없어 LB 에는 영향이 적고, LB 에서도 kumo 가 Otto·GiveMeSomeCredit 에서 1위.
- **LLM 피처는 이득 없음**: LLM 객관식 확률을 GBDT 피처로 붙여도 변화 없음.

## 대회별 결과

### Restaurant Revenue

137행, 회귀, 42 피처. 지표: RMSE ↓ (원 스케일로 평가 → 타깃 변환 없이 학습).

5-fold (2026-10-23, Databricks. 시간은 GBDT CPU 4코어 · TFM H100 이라 직접 비교 불가)

| 모델 | RMSE | ± fold std | 시간 |
|---|---|---|---|
| **kumo** | **2,426,650** | 643,737 | 112s |
| tabicl | 2,439,687 | 643,227 | 38s |
| tabpfn_v2 | 2,441,738 | 746,207 | 95s |
| catboost | 2,640,319 | 606,745 | 14s |
| lgbm | 2,650,611 | 639,363 | 13s |
| xgb | 2,828,951 | 406,260 | 28s |

- TFM 3개가 GBDT 3개보다 모두 나음. 단 fold 하나가 27행이라 차이(약 20만)가 fold std(40~75만)보다 작음 → 순위 확정 불가.

리더보드 (2026-10-23, 2,260팀): **tabpfn_v2 345위 (상위 15.3%)** · public 1.684M / private 1.876M. GBDT 최고 catboost 1,239위 (54.8%).
CV 3위 tabpfn_v2 가 public 1위 — 테스트 10만 행 중 채점 행이 적어 노이즈 큼. private 로는 tabpfn_v2 < lgbm < kumo.

### Titanic

891행, 이진 분류, 12 피처. 지표: accuracy ↑

5-fold (2026-10-20, Kaggle)

| 모델 | Accuracy | ± fold std | AUC | 시간 |
|---|---|---|---|---|
| **exaone** | **0.8462** | 0.015 | **0.8934** | 30s |
| tabicl | 0.8440 | 0.014 | 0.8878 | 9s |
| limix2m | 0.8440 | 0.016 | 0.8839 | 19s |
| catboost | 0.8418 | 0.019 | 0.8749 | 21s |
| xgb | 0.8384 | 0.016 | 0.8839 | 3s |
| tabpfn_v2 | 0.8350 | 0.018 | 0.8755 | 14s |
| lgbm | 0.8339 | 0.016 | 0.8821 | 10s |
| mitra_ft | 0.8339 | 0.019 | 0.8825 | 709s |
| kumo | 0.8328 | 0.018 | 0.8888 | 24s |
| mitra | 0.8294 | 0.023 | 0.8830 | 59s |

- 891행이라 fold std(±1.5~1.9%p)가 모델 간 차이(최대 1.3%p)보다 큼 → accuracy 순위는 의미 없음.
- AUC 도 exaone 이 가장 높음 (0.893).

리더보드 (2026-10-21): exaone public 0.77511 → 점수 기준 4,623 / 11,490 (동점자가 많아 Kaggle 표시 순위는 제출 당시 6,994위).
CV 와 LB 차이가 7%p — 418행 테스트라 변동이 크고, CV 1위 차이도 오차 범위였음.

### House Prices

1,460행, 회귀, 83 피처 중 범주형 44. 지표: log1p(SalePrice) RMSE ↓

5-fold (2026-10-19, Kaggle. limix2m·exaone·mitra 계열은 2026-10-21 재실행)

| 모델 | RMSE | ± fold std | MAE | 시간 |
|---|---|---|---|---|
| **tabpfn_v2** | **0.1151** | 0.016 | 0.0713 | 75s |
| kumo | 0.1191 | 0.023 | 0.0703 | 66s |
| tabicl | 0.1242 | 0.021 | 0.0767 | 21s |
| catboost | 0.1250 | 0.016 | 0.0796 | 154s |
| mitra_ft | 0.1263 | 0.028 | 0.0759 | 7,994s |
| exaone | 0.1275 | 0.032 | 0.0721 | 204s |
| lgbm | 0.1288 | 0.015 | 0.0842 | 21s |
| mitra | 0.1289 | 0.031 | 0.0764 | 259s |
| xgb | 0.1306 | 0.013 | 0.0842 | 14s |
| limix2m | 0.1311 | 0.034 | 0.0741 | 58s |

- TabPFN v2 가 1위, Kumo 가 2위.
- 재실행 모델은 RMSE 는 중위권이지만 MAE 는 exaone(0.0721)이 상위권 → 대부분은 잘 맞히고 일부 이상치 집에서 크게 틀림 (fold std 0.03 안팎으로 큼).
- mitra_ft 는 fine-tuning 으로 mitra 대비 RMSE 0.003 개선, 대신 시간은 30배 (7,994s).
- 상위 모델 간 차이가 fold std(0.013~0.023)보다 작은 경우가 많아 순위 확정은 어려움.
- 1-fold 동작 확인 (참고): tabpfn_v2 0.1177 / tabicl 0.1269 / catboost 0.1284 / lgbm 0.1359 / xgb 0.1363

리더보드 (2026-10-19 제출, 순위는 `km lb` 집계)

| 모델 | CV RMSE | Public LB | 순위 |
|---|---|---|---|
| **kumo** | 0.1191 | **0.10895** | 79 / 4,372 (상위 1.8%) |
| tabpfn_v2 | 0.1151 | 0.11094 | 91 / 4,372 (상위 2.1%) |
| catboost | 0.1250 | 0.12199 | 733 / 4,372 (상위 16.8%) |

- 튜닝 없는 TFM 단일 모델이 상위 2% 안. GBDT 기본 설정보다 훨씬 위.
- CV 에서는 tabpfn_v2 > kumo 였지만 LB 에서는 뒤집힘 → 두 모델 차이는 오차 범위로 봐야 함.
- 이 대회 LB 최상위에는 0.0 점(정답 유출) 제출이 섞여 있어 실제 순위는 이보다 좋은 편.

### Bioresponse

3,751행, 이진 분류, 1,776 피처. 지표: logloss ↓

5-fold 는 아직 없음. fold 0 결과(H100, TFM 은 LightGBM 중요도 상위 500 피처)는 [TFM 추론 속도](#tfm-추론-속도) 참고:
kumo 0.3942 / tabicl 0.3981 / tabpfn_v2 0.4400.

### Mercedes-Benz

4,209행, 회귀, 364 피처 중 범주형 8. 지표: R² ↑

5-fold (2026-10-23, Databricks. 시간은 GBDT CPU 4코어 · TFM H100 이라 직접 비교 불가)

| 모델 | R² | ± fold std | 시간 |
|---|---|---|---|
| **kumo** | **0.5742** | 0.065 | 49s |
| tabpfn_v2 | 0.5717 | 0.064 | 88s |
| tabicl | 0.5673 | 0.070 | 45s |
| catboost | 0.5564 | 0.063 | 101s |
| lgbm | 0.5127 | 0.062 | 65s |
| xgb | 0.4871 | 0.057 | 70s |

- 적은 행 + 많은 피처에서 TFM 우위. TFM 끼리, catboost 와는 fold std 안. lgbm·xgb 와는 0.05 이상 차이.

리더보드 (2026-10-23, 3,825팀): **kumo 1,801위 (상위 47.1%)** · public 0.5573 / private 0.5546. GBDT 최고 catboost 2,713위 (70.9%).

### Spaceship Titanic

8,693행, 이진 분류, 17 피처. 지표: accuracy ↑

5-fold (2026-10-19, Kaggle)

| 모델 | Accuracy | ± fold std | AUC | Logloss | 시간 |
|---|---|---|---|---|---|
| **kumo** | **0.8237** | 0.007 | **0.9192** | **0.3488** | 273s |
| exaone@knn1000 | 0.8151 | 0.009 | 0.9080 | 0.3721 | 7,486s |
| tabpfn_v2 | 0.8135 | 0.007 | 0.9056 | 0.3764 | 92s |
| mitra_ft | 0.8130 | 0.002 | 0.9030 | 0.3819 | 7,587s |
| limix2m | 0.8128 | 0.009 | 0.9039 | 0.3810 | 80s |
| catboost | 0.8125 | 0.008 | 0.9048 | 0.3825 | 66s |
| tabicl | 0.8123 | 0.011 | 0.9038 | 0.3815 | 58s |
| lgbm | 0.8102 | 0.006 | 0.9044 | 0.3848 | 16s |
| xgb | 0.8096 | 0.008 | 0.9045 | 0.3824 | 9s |
| mitra | 0.8059 | 0.006 | 0.8959 | 0.3988 | 265s |

- Kumo 가 accuracy·AUC·logloss 모두 1위.
- Kumo 를 빼면 TFM 과 GBDT 는 0.81 근처에서 오차 범위 안.
- exaone 은 전체 컨텍스트로는 메모리 부족 → kNN 컨텍스트 1000행(`@knn1000_g8_imp`, [컨텍스트 선택](#컨텍스트-선택))으로 실행.

1-fold 동작 확인 (참고, fold 하나 ≈1,740행이라 accuracy 표준오차 ±0.9%p)

| 모델 | Accuracy | AUC | Logloss | 시간 |
|---|---|---|---|---|
| tabicl | 0.8252 | 0.9090 | 0.3715 | 10.5s |
| catboost | 0.8229 | 0.9110 | 0.3698 | 15.4s |
| tabpfn_v2 | 0.8212 | 0.9108 | 0.3646 | 28.1s |
| xgb | 0.8212 | 0.9080 | 0.3755 | 2.4s |
| lgbm | 0.8189 | 0.9066 | 0.3806 | 9.5s |

리더보드 (2026-10-21 제출, 1,746팀, `km lb` 집계): **exaone@knn1000 103위 (상위 5.9%)** · public 0.81014.

### Playground S4E2

20,758행, 7-class 분류, 17 피처. 지표: accuracy ↑

5-fold (2026-10-20, Kaggle T4 ×2 병렬. `@knn1000` 은 2026-10-21)

| 모델 | Accuracy | ± fold std | Logloss | 시간 |
|---|---|---|---|---|
| **tabicl** | **0.9090** | 0.005 | **0.2644** | 137s |
| kumo@knn1000 | 0.9088 | 0.005 | 0.2677 | 10,854s |
| exaone@knn1000 | 0.9087 | 0.004 | 0.2687 | 10,890s |
| xgb | 0.9077 | 0.003 | 0.2686 | 390s |
| tabpfn_v2@knn1000 | 0.9072 | 0.006 | 0.2905 | 5,573s |
| lgbm | 0.9067 | 0.003 | 0.3009 | 224s |
| catboost | 0.9042 | 0.005 | 0.2782 | 954s |
| mitra | 0.8980 | 0.003 | 0.3021 | 1,381s |
| limix2m | 0.8971 | 0.006 | 0.3050 | 311s |
| mitra_ft | 0.8969 | 0.007 | 0.2973 | 11,081s |

- TabICL 이 1위이면서 가장 빠름 (137s). 비용 대비로는 TabICL 이 압도적.
- 상위 모델 차이는 fold std 수준.
- tabpfn_v2·exaone·kumo 는 전체 컨텍스트로 실패 (tabpfn_v2 는 당시 1만 행 공식 한도 초과로 거부 — 이후 `ignore_pretraining_limits=True` 로
  한도 검사는 풀었지만 전체 컨텍스트 5-fold 는 다시 안 돌림, exaone 은 math 어텐션이 445GiB 요청,
  kumo 는 13.7GiB 요청으로 T4 메모리 초과) → `@knn1000` (쿼리 16개 묶음마다 가장 가까운 1000행만 컨텍스트)으로 모두 GBDT 수준 이상.

리더보드 (종료된 대회, 늦은 제출. 순위는 `km lb` 집계의 public 점수 기준 위치, 3,590팀)

| 모델 | CV | Public LB | Private LB | 순위 |
|---|---|---|---|---|
| **kumo@knn1000** | 0.9088 | **0.91076** | **0.90959** | 1,452 / 3,590 (상위 40.4%) |
| exaone@knn1000 | 0.9087 | **0.91076** | 0.90688 | 1,452 / 3,590 (상위 40.4%) |
| tabpfn_v2@knn1000 | 0.9072 | 0.90787 | 0.90634 | 1,843 / 3,590 (상위 51.3%) |

- CV 순서와 LB 순서가 대체로 일치.

### Amazon Employee Access

32,769행, 이진 분류, 9 피처 (모두 범주 ID). 지표: AUC ↑

5-fold (2026-10-22, Databricks A10G, TFM 컨텍스트 1만 행 무작위. `kumo (ctx 전체)` 는 2026-10-26 H100 재실행, fold 학습 2.6만 행 전부)

| 모델 | AUC | ± fold std | Logloss | 시간 |
|---|---|---|---|---|
| **catboost** | **0.8957** | 0.008 | **0.1374** | 139s |
| kumo (ctx 전체) | 0.8795 | 0.009 | 0.1414 | 96s |
| xgb | 0.8734 | 0.010 | 0.2163 | 52s |
| lgbm | 0.8631 | 0.012 | 0.1672 | 14s |
| kumo | 0.8274 | 0.014 | 0.1644 | 629s |
| tabicl | 0.8054 | 0.009 | 0.1733 | 52s |
| tabpfn_v2 | 0.7781 | 0.009 | 0.1894 | 131s |

- 범주 ID 만 있는 데이터에서 catboost 가 앞섬. 컨텍스트를 전체로 풀면 kumo 가 0.8274 → 0.8795 로 오르지만, catboost 와 AUC 0.016 차이 (fold std 의 약 2배)가 남음.
- 남은 격차의 원인은 확인 안 함. 9개 열이 모두 범주 ID 라 범주 처리 방식 차이가 원인일 것으로 추정 (미검증). 외부 BeyondArena 고카디널리티 부분집합에서도 CatBoost 가 강함.
- kumo (ctx 전체) 시간 96s 는 H100, 1만 행 kumo 629s 는 A10G 라 환경이 달라 직접 비교 안 됨.

리더보드 (2026-10-22, 1,689팀): **catboost 64위 (상위 3.8%)** · public 0.91582 / private 0.91100. xgb 529위 (31.3%).
kumo (ctx 전체) 502위 (29.7%) · public 0.89432 / private 0.89114 (2026-10-26).

### Otto

61,878행, 9-class 분류, 93 피처. 지표: logloss ↓

5-fold (2026-10-23, Databricks, TFM 컨텍스트 1만 행 무작위. 시간은 GBDT CPU 4코어 · TFM H100 이라 직접 비교 불가. `kumo (ctx 전체)` 는 2026-10-26 H100 재실행, fold 학습 약 5만 행 전부)

| 모델 | logloss | ± fold std | 시간 |
|---|---|---|---|
| **kumo (ctx 전체)** | **0.3774** | 0.0033 | 302s |
| lgbm | 0.4648 | 0.0059 | 842s |
| kumo | 0.4665 | 0.0034 | 150s |
| tabicl | 0.4974 | 0.0047 | 97s |
| xgb | 0.5002 | 0.0056 | 456s |
| catboost | 0.5382 | 0.0066 | 960s |
| tabpfn_v2 | 0.5641 | 0.0060 | 653s |

- 1만 행 kumo 는 lgbm 과 사실상 동률 (차이 0.0017 < fold std), 1만 행 기준 TFM 간 편차도 큼 (kumo 0.467 ↔ tabpfn_v2 0.564).
- 상한을 풀면 kumo 0.3774 로 lgbm 보다 0.087 낮음 (lgbm fold std 0.006 의 약 15배).

리더보드 (2026-10-23, 3,510팀): **kumo 420위 (상위 12.0%)** · public 0.4348 / private 0.4361. GBDT 최고 lgbm 726위 (20.7%).
CV 에서는 동률이었지만 LB 에서는 kumo 가 확실히 앞섬 (0.435 vs 0.451).
kumo (ctx 전체) **1위 (<0.1%)** · public 0.36714 / private 0.36924 (2026-10-26, km lb 집계). 종료 대회라 점수 기준 위치이며, 2015 대회 우승 점수 0.38242 보다도 낮음.

### GiveMeSomeCredit

150,000행, 이진 분류, 10 피처. 지표: AUC ↑

5-fold (2026-10-22, Databricks A10G, TFM 컨텍스트 1만 행 무작위. `kumo (ctx 전체)` 는 2026-10-26 H100 재실행, fold 학습 12만 행 전부)

| 모델 | AUC | ± fold std | Logloss | 시간 |
|---|---|---|---|---|
| **kumo (ctx 전체)** | **0.8694** | 0.0034 | **0.1757** | 500s |
| catboost | 0.8660 | 0.0031 | 0.1777 | 59s |
| lgbm | 0.8653 | 0.0037 | 0.1775 | 14s |
| kumo | 0.8652 | 0.0032 | 0.1778 | 1,247s |
| xgb | 0.8647 | 0.0037 | 0.1780 | 13s |
| tabicl | 0.8631 | 0.0033 | 0.1799 | 102s |
| tabpfn_v2 | 0.8582 | 0.0034 | 0.1813 | 284s |

- 1만 행 (학습의 8%) 에서는 catboost, lgbm, kumo, xgb 가 fold std 안 동률.
- 상한을 풀면 (fold 학습 12만 행 전부) kumo 0.8694 로 catboost 를 0.0034 앞섬 — fold std 수준이라 확정은 아님. LB 에서는 차이가 더 큼 (아래).
- TFM 시간은 속도 개선 전 A10G 기준. H100 + 코드 개선 후 fold 당 10~36초 ([TFM 추론 속도](#tfm-추론-속도)).

리더보드 (2026-10-22, 926팀): **catboost 27위 (상위 2.9%)** · public 0.86271 / private 0.86886. lgbm 64위 (6.9%).
kumo (ctx 전체) **1위 (0.1%)** · public 0.86528 / private 0.87112 (2026-10-26, km lb 집계).

### Allstate Claims Severity

188,318행, 회귀, 130 피처 중 범주형 116. 지표: MAE ↓ (원 스케일로 평가 → 타깃 변환 없이 학습).

5-fold (2026-10-23, Databricks, TFM 컨텍스트 1만 행 무작위. 시간은 GBDT CPU 4코어 · TFM H100 이라 직접 비교 불가. `kumo (ctx 전체)` 는 2026-10-26 H100 재실행, 중앙값 예측, fold 학습 15만 행 전부)

| 모델 | MAE | ± fold std | 시간 |
|---|---|---|---|
| **kumo (ctx 전체)** | **1,114.2** | 7.8 | 959s |
| kumo | 1,181.2 | 6.2 | 212s |
| lgbm | 1,188.3 | 5.7 | 227s |
| catboost | 1,188.4 | 6.3 | 4,764s |
| xgb | 1,188.5 | 6.0 | 181s |
| tabicl | 1,198.6 | 5.2 | 138s |
| tabpfn_v2 | 1,209.5 | 6.3 | 677s |

- 1만 행 (학습의 7%) 에서도 kumo 가 GBDT 보다 MAE 7 낮음 (fold std 수준). 상한을 풀면 1,114.2 로 lgbm 보다 74 낮음 (fold std 의 10배 안팎).
- 재실행 1차는 분위수 999개의 평균으로 예측해 MAE 1,144.0 (± 6.4, 910s) 이었음. MAE 는 중앙값이 맞아서 `km/models/tfm.py` 의 `make_kumo(metric=...)` 를 고쳐 중앙값으로 다시 돌림 (위 행).
- catboost 는 범주형 116개 때문에 CPU 4코어에서 4,764s.

리더보드 (2026-10-23, 3,048팀): kumo (1만 행) **1,919위 (상위 63.0%)** · public 1,154.0 / private 1,162.6. GBDT 최고 lgbm 2,054위 (67.4%).
kumo (ctx 전체, 중앙값) **1위 (<0.1%)** · public 1,095.69 / private 1,108.12 (2026-10-26, km lb 집계). 평균 예측 제출(1차)은 public 1,126.62 / private 1,135.67, 1,555위 (51.0%).
종료 대회라 점수 기준 위치. 기존 "단일 모델은 상위권에 못 미침" 설명은 1만 행 kumo 기준이었음.

### Playground S6E1

630,000행, 회귀, 11 피처 중 범주형 7. 지표: RMSE ↓

5-fold (2026-10-23, Databricks, TFM 컨텍스트 1만 행 무작위. 시간은 GBDT CPU 4코어 · TFM H100 이라 직접 비교 불가. `kumo (ctx 20만)` 는 2026-10-26 H100 재실행, 20만 행 무작위 — fold 학습의 약 40%)

| 모델 | RMSE | ± fold std | 시간 |
|---|---|---|---|
| **kumo (ctx 20만)** | **8.6767** | 0.0164 | 1,952s |
| xgb | 8.7736 | 0.0126 | 172s |
| lgbm | 8.7847 | 0.0125 | 284s |
| catboost | 8.7924 | 0.0123 | 2,158s |
| kumo | 8.8562 | 0.0118 | 369s |
| tabicl | 8.8927 | 0.0107 | 71s |
| tabpfn_v2 | 8.8985 | 0.0076 | 238s |

- 1만 행 kumo 는 GBDT 3개보다 모두 나빴음 (catboost ↔ kumo 0.064 = fold std 의 5배). 1만 행은 fold 학습의 2% 라 데이터를 거의 못 씀.
- 20만 행으로 올리자 kumo 8.6767 로 xgb 를 0.097 앞섬 (fold std 의 6배 이상). 이 대회의 GBDT 우위는 TFM 자체보다 컨텍스트 상한 때문이었음.

리더보드 (2026-10-23, 4,320팀): xgb 2,707위 (상위 62.7%) · public 8.7440 / private 8.7678. 1만 행 kumo 3,449위 (79.8%).
kumo (ctx 20만) **764위 (상위 17.7%)** · public 8.63047 / private 8.6608 (2026-10-26, km lb 집계). 이 대회 최고 LB.

### Playground S6E10

699,635행, 이진 분류, 22 피처, 진행 중 대회. 지표: AUC ↑ (제출이 확률값이라 AUC 로 추정).
TFM 은 fold 마다 학습 데이터에서 1만 행만 무작위로 뽑아 컨텍스트로 사용 (`tfm_max_rows: 10000`), GBDT 는 전체(fold 당 ~56만 행) 사용.
kumo 만 2026-10-26 에 컨텍스트 20만 행 무작위로 다시 돌림 (TabArena Kumo 래퍼 기본 상한, fold 학습의 약 36%).

5-fold (2026-10-20, Kaggle T4 ×2 병렬. `kumo (ctx 20만)` 는 2026-10-26 Databricks H100 이라 시간은 직접 비교 불가)

| 모델 | AUC | ± fold std | Accuracy | Logloss | 시간 |
|---|---|---|---|---|---|
| **kumo (ctx 20만)** | **0.95880** | 0.0006 | 0.9270 | 0.2254 | 2,131s |
| lgbm | 0.95838 | 0.0006 | 0.9256 | 0.2286 | 556s |
| catboost | 0.95820 | 0.0007 | 0.9255 | 0.2292 | 1,986s |
| xgb | 0.95815 | 0.0007 | 0.9253 | 0.2295 | 819s |
| limix2m | 0.95442 | 0.0009 | 0.9209 | 0.2411 | 2,400s |
| tabpfn_v2 | 0.95411 | 0.0005 | 0.9208 | 0.2422 | 5,975s |
| mitra | 0.95195 | 0.0008 | 0.9173 | 0.2486 | 14,618s |
| tabicl | 메모리 부족으로 실패 (이후 예측을 2만 행 단위로 나눠 해결, 5-fold 재실행은 보류) | | | | |
| kumo | 2026-10-20 메모리 부족 실패 → 위 `kumo (ctx 20만)` 행으로 대체 | | | | |

- 1만 행 (전체의 1.8%) 에서는 GBDT 가 확실히 앞섬 (AUC +0.004, fold std 의 5배 이상). 이 차이는 컨텍스트 상한 탓이었음.
- 20만 행 kumo 는 0.95880 으로 lgbm 0.95838 과 0.0004 차이 — fold std 이내 동률. LB 에서도 kumo 0.95835 > lgbm 0.95792 (아래). 전체 56만 행에서의 우열은 미확인.
- TFM 은 예측 행이 많아 느림 (fold 당 검증 14만 + 테스트 30만 행). Mitra 는 14,618s.
- 컨텍스트를 kNN·무작위로 키워 격차를 줄이는 실험은 [컨텍스트 선택 › S6E10](#s6e10) 참고 (tabicl 무작위 20만 행에서 lgbm 과 −0.0008).

리더보드 (public LB, 진행 중, 순위는 `km lb` 집계 — 팀 수가 2,649 → 3,071 로 늘어 이전 순위와 다름)

| 모델 | CV | Public LB | 순위 |
|---|---|---|---|
| **kumo (ctx 20만)** | 0.9588 | **0.95835** | 2,009 / 3,071 (상위 65.4%) |
| lgbm | 0.9584 | 0.95792 | 2,470 / 3,071 (상위 80.4%) |

## 주제별 실험

### 컨텍스트 선택

가설: 쿼리와 비슷한 학습 샘플 k개만 컨텍스트로 넣어도 전체 데이터만큼 추론할 수 있다 → 대용량에서도 TFM 사용 가능.
구현: `km/models/context.py` (브랜치 `knn-context` 에서 시작). 모델 이름 뒤에 `@<spec>` 을 붙여 사용 (예: `tabicl@knn2000_g256_imp_bal20`).

| spec | 뜻 |
|---|---|
| `knnK` / `randK` / `partK` | 쿼리와 가까운 K행 / 무작위 K행 / k-means 영역 중심에 가까운 K행 (쿼리는 가장 가까운 영역으로) |
| `gN` | knn 에서 컨텍스트 하나를 공유하는 쿼리 묶음 크기 (클수록 빠르고 거칠어짐) |
| `imp` / `leaf` | 거리 공간: LightGBM 중요도 가중 / LightGBM leaf 임베딩 (기본은 표준화 수치 + 원핫 범주) |
| `full` | 학습 fold 전체 (래퍼 없이 돌린 것과 같은 기준선, `results.csv` 의 `@full` 행) |
| `mixP` | 컨텍스트의 P% 를 무작위 행으로 채움 (이웃 쏠림 완화) |
| `balP` | 컨텍스트의 P% 를 클래스 균형 코어셋으로 채움 |

실행: Spaceship·S4E2 와 S6E10 2026-10-22 실행은 Colab 무료 T4 (`km colab`), S6E10 2026-10-21 실행은 Kaggle T4. 모두 테스트 예측 생략.

#### Spaceship

5-fold (2026-10-20, 학습 fold 약 6,950행) — accuracy / AUC / logloss

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
- 비용: kNN 은 쿼리 8개 묶음마다 재-fit → fold 당 180~450s (전체는 수십 초). 이점은 전체가 컨텍스트에 안 들어가는 대용량에서만 의미 있음.
- 1-fold 스모크(fold 0): 거리 공간 imp(LightGBM 중요도 가중)가 raw 보다 좋음 (TabPFN 0.8212 vs 0.8194).

#### S4E2

학습 fold 약 16,600행, 7클래스. 5-fold (2026-10-21) — accuracy / logloss

| 컨텍스트 | TabICL | TabPFN v2 |
|---|---|---|
| 전체 | 0.9089 / 0.2644 (60s) | 불가 (당시 1만 행 한도 초과로 거부, 한도 해제 후 재실행 안 함) |
| 무작위 1000 | 0.8872 / 0.3369 | 0.8818 / 0.3493 |
| kNN 1000 (g16, imp) | 0.9064 / 0.2750 (1,799s) | 0.9055 / 0.2918 (1,303s) |

같은 fold 의 GBDT: xgb 0.9077, lgbm 0.9067, catboost 0.9042. fold std 0.004~0.006.

- 컨텍스트 6%(1000/16,600)로 무작위 대비 +1.9~2.4%p, TabICL 은 전체와 −0.25%p (fold std 이내).
- TabPFN v2 는 전체 데이터를 못 넣는 크기인데, kNN 1000 으로 GBDT 수준(0.9055)에 도달 → 가설의 실용적 의미가 처음 드러남.
- 다수 클래스 비율 0.68 (7클래스인데도 이웃이 한 클래스로 크게 쏠림). logloss 는 전체보다 약간 나쁨.

검색 공간·구성 방식 비교 (2026-10-21, TabICL, 컨텍스트 1000행) — accuracy / logloss / 총 시간

| 방식 | 결과 |
|---|---|
| 전체 (기준) | 0.9089 / 0.2644 / 60s |
| kNN imp (기존) | 0.9064 / 0.2750 / 1,799s |
| kNN leaf | 0.9073 / 0.2753 / 2,034s |
| **kNN imp + bal20** | **0.9076** / 0.2774 / 1,841s (다수 클래스 비율 0.68 → 0.59) |
| part imp (영역 17개) | 0.8945 / 0.3182 / 139s |
| part leaf | 0.8816 / 0.3556 / 130s |
| 무작위 (기준) | 0.8872 / 0.3369 / 14s |

TabPFN v2 part leaf: 0.8752 / 0.4040 (kNN imp 0.9055 대비 크게 낮음).

- leaf 공간·bal 코어셋은 kNN imp 대비 +0.1%p 수준 (fold std 0.4%p 이내, 유의하지 않음). 전체와 차이는 0.13%p 까지 줄어듦.
- bal 은 이웃 쏠림을 줄였지만 logloss 는 개선되지 않음.
- **영역 분할(part)은 실패**: 13배 빠르지만 무작위와 비슷하거나 더 나쁨. 영역 중심 하나에 가까운 1000행은 영역 가장자리 쿼리에 맞지 않음 → "쿼리마다 가까움"이 핵심.

묶음 크기 (2026-10-21, TabICL, imp + bal20) — accuracy / logloss / 총 시간

| 설정 | 결과 |
|---|---|
| k1000 g16 | 0.9076 / 0.2774 / 1,841s |
| k1000 g64 | 0.9049 / 0.2815 / 490s |
| k1000 g256 | 0.9043 / 0.2877 / 142s |
| **k2000 g256** | **0.9078 / 0.2752 / 137s** |
| 전체 (기준) | 0.9089 / 0.2644 / 60s |

- 묶음을 키우면 조금씩 떨어지지만(g16→g256 −0.3%p), 컨텍스트를 2000행으로 늘리면 회복 이상 → g16 보다 좋고 13배 빠름.
- k2000 g256 은 전체와 0.11%p 차이, 총 시간도 전체의 2.3배 수준까지 줄어듦 → S6E10 에 쓸 설정.

#### S6E10

학습 fold 약 56만 행, fold 0 만, 테스트 예측 생략. 전체 컨텍스트는 TFM 에 못 넣는 규모라 기준은 무작위 1만 행.

kNN 탐색 (2026-10-21) — AUC

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
- 5-fold + 테스트 예측이면 모델당 약 12,600s (fold 당 검증 14만 + 테스트 30만 행).

TFM 격차 원인 분석 (2026-10-21) — AUC

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

S4E2 에서 고른 설정(g256 + bal20) 적용 (2026-10-22) — AUC / accuracy / logloss / 시간

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

#### 정리

- 전체 데이터가 컨텍스트에 들어가면 kNN 을 쓸 이유가 없음 (전체와 비슷하거나 약간 낮고 훨씬 느림).
- 한도를 넘는 중간 크기(S4E2 1.6만 행)에서는 kNN 1000~2000 으로 전체·GBDT 수준까지 회복.
- 수십만 행(S6E10)에서는 kNN 보다 무작위 대용량 컨텍스트가 더 좋음. fold 0 에서 tabicl·tabpfn_v2 는 GBDT 를 못 넘었고, kumo 는 20만 행 5-fold 에서 GBDT 와 동률 ([S6E10 섹션](#playground-s6e10)).
- 쿼리 단위 지역성이 핵심: 영역 분할(part)은 빠르지만 무작위 수준. 비용은 묶음 크기 g 를 키우고 k 를 늘리는 쪽으로 줄임.

### 외부 벤치마크 대조

TabArena·BeyondArena 리더보드 CSV (HF space `TabArena/leaderboard`, 2026-10-25 갱신) 기준. 외부 수치는 재현하지 않았고, 외부 GBDT 는 튜닝된 설정이라 이 문서의 기본 설정 결과와 직접 비교 불가.

- 10만 행 이하: TabArena 에서도 FM 이 대체로 우세.
- 10만 행 초과 (BeyondArena, 6개 데이터셋, 신뢰구간 넓음): FM 8개 중 2개만 최고 트리 모델보다 나음. Kumo 는 Elo 가 가장 높지만 평균 improvability 는 튜닝된 CatBoost 보다 나쁨 (4.2% vs 2.7%).
- 이 저장소의 모델 순서 (kumo > … > tabicl > tabpfn_v2) 는 TabArena 순서와 같음.

### 의미 기반 결정 모델

LLM 객관식 · SAP-RPT (2026-10-22, 브랜치 `decision-models`).
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
| Spaceship (acc ↑) | 0.8102 | 0.8237 kumo | 0.8051 | 0.7144 | 0.7208 | 0.8123 | 0.8081 |
| Spaceship (AUC ↑) | 0.9044 | 0.9192 kumo | 0.8954 | 0.7278 | 0.7421 | 0.9043 | 0.9034 |
| S4E2 (acc ↑) | 0.9067 | 0.9090 tabicl | 0.8984 | 0.8127 | 0.8086 | 0.9028 | 0.9049 |
| S4E2 (logloss ↓) | 0.3009 | 0.2644 tabicl | 0.2991 | 0.5679 | 0.5720 | 0.3224 | 0.3194 |
| House Prices (RMSE ↓) | 0.1288 | 0.1151 tabpfn_v2 | 0.1217 | — | — | — | — |

5-fold 시간: llm 은 Titanic 1,185s(로컬), Spaceship 1,490s, S4E2 4,157s / llm8b 는 2,419s, 6,714s /
sap_rpt 는 House 49s, Spaceship 245s, S4E2 649s. llm_lgbm 은 llm 채점 결과를 재사용해 GBDT 시간만 듦
(측정 당시는 같은 프로세스 캐시, 지금은 모델별 프로세스라 디스크 캐시 `~/.cache/km/llm`).

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

### TFM 추론 속도

2026-10-22, Databricks, fold 0, 테스트 예측 포함 — 초.
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

- "이전" 중 ~ 는 5-fold 총 시간 / 5.
- 코드 개선: fold 당 검증·테스트를 한 번에 예측 (컨텍스트 계산 1회), TabPFN 컨텍스트 KV 캐시(GPU 에 들어갈 때만),
  Kumo 는 sdm `fit()`/`predict()` 로 컨텍스트 1회 인코딩, 모델마다 새 프로세스 (tabicl -9 해결).
  lgbm / tabpfn_v2 / tabicl 은 합쳐 예측해도 출력이 완전히 같음 (로컬 확인).
- 넓은 데이터는 피처 수가 결정적: TabPFN v2 프리셋은 피처 서브샘플링이 없어 1,776개 × 8 estimator. `tfm_max_features: 500`
  (LightGBM 중요도 상위)로 14배 빨라지고 점수도 좋아짐 (tabpfn_v2 0.4486→0.4400, tabicl 0.4276→0.3981).
- bioresponse tabpfn_v2 의 fit 380s 는 CPU 전처리. 로컬 맥은 같은 버전으로 12s → 원인 미확인, 피처 500개면 8s 라 보류.
- H100 에서는 OOM 경고 0회. kumo_e4 는 AUC 차이 0.00004 로 2.4배 빠름 (fold 하나, 대회 하나 기준).

### Databricks 실행 준비 시간

2026-10-23, H100, GiveMeSomeCredit fold 0 tabpfn_v2·tabicl·kumo, 테스트 예측 생략 — 초.
fold 당 모델 시간이 10~30초까지 줄어 실행 준비(pip 설치·가중치 다운로드·이미지)가 더 커짐 → 가중치 Volume 캐시와 패키지 이미지.

| 방식 | 패키지 준비 | km run | 러너 안 | 러너 밖 (GPU 할당·환경/이미지 준비) | job 전체 |
|---|---|---|---|---|---|
| pip, 가중치 캐시 없음 (e2-demo) | 80 | 61 | 151 | 62 | 213 |
| pip, 가중치 캐시 (e2-demo) | 78 | 38 | 122 | 60 | 182 |
| 이미지 km-air:v4, 새 태그 첫 실행 (FEVM) | 2 | 44 | 53 | 170 | 223 |
| 이미지 km-air:v4, 두 번째 (FEVM) | 6 | 86 ※ | 99 | 54 | **153** |

- 가중치 캐시: km run 61→38s (kumo 33→15s, 가중치 다운로드 제거), 복원 3s.
- 이미지: pip 78s 제거. 노드에 이미지가 있으면 러너 밖 시간도 pip 방식과 비슷(54 vs 60s). 새 태그·새 노드는 17.5GB 를 받느라 +약 120s.
- ※ 두 번째 실행은 v3·v4 모두 tabpfn_v2 가 37~40s (첫 실행 11s) — 노드 차이로 추정, 원인 미확인.
- 이미지는 torch 2.13.0 (autogluon 과 함께 해결된 버전) → tabpfn_v2 AUC 0.85717 → 0.85711 로 소폭 변화.
- 워크스페이스 권한·베이스 이미지 관련 메모는 [docs/known-issues.md](docs/known-issues.md#databricks).

## 다음 할 일

- [ ] tabicl·tabpfn_v2 컨텍스트 상한 해제 재실행 (진행 중, 다른 에이전트 실행): Amazon·Otto·GiveMeSomeCredit·Allstate·S6E1·S6E10. kumo 와 같은 조건이어야 TFM 간 CV 순위 비교가 공정. 결과 나오면 요약·대회별 표 갱신
- [ ] S6E10 tabicl 대용량 무작위 컨텍스트 5-fold + 제출 (위 재실행 항목에 포함, rand100000 이면 fold 당 약 1,800s + 테스트 30만 행 예측). kNN 5-fold 는 무작위 대용량이 더 좋아 대체함
- [ ] S4E2 전체 컨텍스트 (fold 학습 약 1.66만 행): tabpfn_v2 는 한도 해제 후 전체로, kumo·exaone 은 T4 메모리로 못 돌렸던 전체 컨텍스트를 H100 에서 시도 → kNN 1000 과 비교
- [ ] S6E10 kumo 컨텍스트 전체 (56만 행): 20만 행 대비 추가 이득과 시간 확인
- [ ] Amazon 격차 원인 검증 (kumo 0.8795 ↔ catboost 0.8957, 범주 ID 처리 가설)
- [ ] `km board` 재생성 — 지금 docs/leaderboard.md 는 Allstate 평균 예측 제출(LB 1,126.62) 을 보여줌
- [ ] Bioresponse 5-fold + 제출 (TFM 은 `tfm_max_features: 500`)
- [ ] sap_rpt8k (컨텍스트 8192) H100 에서 5-fold — 컨텍스트를 키우면 TFM 상위권에 붙는지
- [ ] TFM + GBDT 스태킹
- [ ] 패키지 버전 고정 및 실행 시 버전 기록
