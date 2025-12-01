# WBCD-Model-Comparison
# 🎀 유방암 진단 유효성 비교 및 특징 중요도 해석 연구 (WBCD 기반)

## ⭐️ 1. 프로젝트 개요 (Project Overview)

### 🧐 문제정의(Problem Identification)
유방암 조기 진단의 중요성에도 불구하고, 기존 세포병리 기반 진단은 검사자의 주관성과 **특징 변수 간의 복잡한 비선형 관계**로 인해 오진 가능성이 존재합니다. 
본 연구는 **정량적 의사결정 지원 시스템 구축**을 목표로, 세포 형태학적 특징을 활용한 머신러닝 모델(SVM, MLP, Random Forest)을 비교 및 해석하여 병리 판독의 보조 도구로 활용 가능한 설명력 있는 모델을 탐색합니다.
이를 통해 본 연구는 **병리 판독의 보조 도구** 로 활용 가능한 모델을 탐색하고 임상의가 수용할 수 있는 **설명력** 을 갖춘 모델을 구축하는 것을 목표로 합니다.

### 🎯 프로젝트 목표 (Project Goal)
1. **모델 유효성 비교**: SVM, MLP, Random Forest 모델의 유방암 진단 성능을 ROC-AUC 및 $K$-Fold CV를 기준으로 안정적으로 비교 분석합니다.
2. **통계적 신뢰도 확보**: Bootstrap CI와 $K$-Fold ROC Curve 시각화를 통해 모델 성능의 일반화 및 안정성을 입증합니다.
3. **특징 중요도 해석**: SHAP(SHapley Additive exPlanations) 분석을 통해 비선형 모델의 예측을 해석하고, 임상적 의미를 부여합니다.
4. **임상 최적 모델 제시**: 높은 성능, 통계적 안정성, 그리고 임상적 해석 가능성을 종합적으로 고려하여 최적 모델을 제안합니다.
---
### 🎯 목표모델 성능 수준 (Target Performance)
* **SVM 모델**: $AUC \ge 0.96$
* **MLP 모델**: $AUC \ge 0.97$
* **rf 모델**: $AUC \ge 0.95$

## 💾 2.데이터 소개 및 전처리 (Data Introduction & Preprocessing)

### 📌 분석 대상 데이터 (WBCD)
* **데이터 출처:** UCI Machine Learning Repository (Wisconsin Breast Cancer Diagnostic Dataset)
* **분석 대상:** 미세침 흡인(FNA) 이미지에서 추출된 세포 핵의 형태적 특징
* **변수 구성:** 총 32개
    * **ID:** 환자 식별 번호 
    * **Diagnosis (타겟):** **M (암, 1)** 또는 **B (양성, 0)**
    * **30개 Feature:** 10가지 세포핵 측정 항목 (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)에 대해 **Mean, Standard Error (SE), Worst** 통계량을 계산하여 구성.
* **샘플 개수:** 총 569개 (양성 357개, 악성 212개)
* **결측치:** 없음
 <img width="2644" height="1669" alt="data_info_summary" src="https://github.com/user-attachments/assets/026b42d8-d712-40ca-aaab-1eb1ec18aba5" />


### 📊 주요 데이터 탐색 결과 (EDA)
#### 1. 타겟 변수 분포
악성(1)과 양성(0) 샘플 비율이 약 2:3으로 **약간의 불균형**은 있지만, 모델 학습을 방해할 정도는 아닙니다.
```python
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    print("\n--- 2. 데이터 샘플 확인 (Head) ---")
    print(df.head())

    print("\n--- 3. 타겟 변수 분포 확인 ---")
```
<img width="800" height="600" alt="target_distribution" src="https://github.com/user-attachments/assets/bd27f403-a8bc-48a3-a445-beedb5755508" />

#### 2. 특징 간 상관관계 (Mean Features)
`radius`, `perimeter`, `area` 등 세포핵의 크기와 관련된 특징들 간에 **다중공선성**이 관찰되었습니다. 따라서 선형 분류기가 아닌 비선형 분류기로 구성하고
**Permutation Importance**로 각 변수의 순수한 영향력을 분석합니다.

* Permutation Importance는 특정 특징의 값을 무작위로 섞었을 때 모델 성능이 얼마나 감소하는지를 측정합니다.
* 모델 종류에 상관없이 적용 가능하며, 다중공선성이 있는 데이터에서도 신뢰도가 높습니다.


```python
features_mean = list(df.columns[1:11]) 
    corr = df[features_mean].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('주요 특징 간 상관관계 히트맵 (Mean Features)')
    plt.savefig("correlation_heatmap.png")
    print("INFO: 'correlation_heatmap.png' ")

```
<img width="1200" height="1000" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/e2bd9218-287f-4c2d-ae43-c2820c1b6555" />


---

## ⚙️ 전처리 요약

### 1. 전처리 (Preprocessing)
* 거리에 민감한 SVM, MLP 모델의 안정적인 학습을 위해  불필요한 열을 제거하고, 타겟 변수를 숫자로 변환한 후, **StandardScaler**를 이용하여 모든 특징을 표준화했습니다. 
* id, $\text{Unnamed: 32}$ 열 제거 및 diagnosis 라벨 인코딩.
* StandardScaler를 이용하여 모든 특징을 표준화 (SVM, MLP 모델 학습 안정화).
* Stratified Sampling 기반으로 학습/테스트 데이터 분리 ($\text{Test size} = 0.2, \text{random state} = 42$).
```python
# 불필요한 열 제거 및 타겟 변수 인코딩
df = df.drop(['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 학습/테스트 데이터 분리 (Stratified Sampling)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 특징 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. SVM 하이퍼파라미터 튜닝
Support Vector Machine (SVM) 모델의 성능을 최적화하기 위해 핵심 하이퍼파라미터인 $C$와 $\gamma$를 튜닝했습니다.

| 파라미터 | 역할 | 탐색 범위 |
| :--- | :--- | :--- |
| **C (Cost)** | 오분류에 대한 **벌칙** (규제 강도) | `[0.1, 1, 10, 100, 1000]` |
| **$\gamma$ (Gamma)** | 하나의 데이터 샘플이 미치는 **영향 범위** (결정 경계의 복잡도) | `[0.1, 0.01, 0.001, 0.0001]` |
| **Kernel** | 데이터를 고차원 공간으로 매핑하는 함수 | `['rbf']` (고정) |

```python
# GridSearchCV를 이용한 SVM 튜닝 코드
param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
grid_search_svm = GridSearchCV(
    estimator=SVC(probability=True, random_state=42),
    param_grid=param_grid_svm,
    cv=5,
    scoring='roc_auc'
)
grid_search_svm.fit(X_train_scaled, y_train)

# 최적 모델 추출
svm_model = grid_search_svm.best_estimator_
```
## 🥇 3. 모델 유효성 검증 및 신뢰도 분석 (Model Validation Framework)
본 연구는 단순한 성능 비교를 넘어, 모델의 통계적 신뢰성과 임상적 활용성을 입증하기 위해 다음과 같은 고급 검증 프레임워크를 적용했습니다.

### 3.1. $K$-Fold 교차 검증 및 안정성 분석
$\text{SVM}$과 $\text{Random Forest}$ 모두 10-Fold CV 결과, 평균 $\text{AUC}$가 높고 표준편차가 낮아($\sigma < 0.02$) 데이터 분할에 관계없이 모델이 일관된 성능을 보이며 높은 일반화 능력을 갖췄음을 확인했습니다.

| 모델 | 평균 AUC ($\mathbf{\mu}$) | 표준편차 ($\mathbf{\sigma}$) | 안정성 평가 |
| :--- | :--- | :--- | :--- |
| **SVM** | $\mathbf{0.9915}$ | $\mathbf{0.0062}$ | **가장 안정적** (낮은 $\sigma$로 일반화 능력 우수) |
| **Random Forest** | $0.9847$ | $0.0147$ | **매우 안정적** |
| **MLP** | (미분석) | (미분석) | - |

<img width="1000" height="800" alt="roc_kfold_Random_Forest" src="https://github.com/user-attachments/assets/ab8f5e54-0965-4e64-9810-b33bca07163a" />
<img width="1000" height="800" alt="roc_kfold_SVM" src="https://github.com/user-attachments/assets/c7bb52cd-306d-4523-a146-906204391a43" />




### 3.2. 임계값 기반 임상 의사결정 지원 테이블 (Random Forest 기준)
예측 확률 $\text{Threshold}$ 변화에 따른 민감도(Sensitivity) 및 특이도(Specificity) 변화를 분석하여, 임상 환경별 최적의 의사결정 기준을 도출합니다.
| Threshold | Sensitivity (악성 놓칠 확률 $\downarrow$) | Specificity (양성을 악성으로 진단할 확률 $\downarrow$) | 임상적 의미 |
| :---: | :---: | :---: | :--- |
| **0.1** | $\mathbf{1.0000}$ | $0.8056$ | **최대 민감도 확보:** 악성(암)을 놓치지 않아야 하는 1차 스크리닝에 적합. |
| **0.3** | $0.9286$ | $0.9722$ | **균형점:** 높은 민감도와 특이도의 적절한 균형. |
| **0.5** | $0.9286$ | $\mathbf{1.0000}$ | **최대 특이도 확보:** 양성을 악성으로 오진하는 경우(False Positive)를 완전히 배제해야 하는 정밀 진단에 적합. |
| **0.7** | $0.8333$ | $\mathbf{1.0000}$ | **고신뢰도 선별:** 극도로 신뢰성 있는 악성 진단만 허용. |
| **0.9** | $0.5952$ | $\mathbf{1.0000}$ | (과도하게 보수적인 임계값) |

## 🥈 4. 비선형 모델 해석 강화 (Explainability & Interpretation)
가장 균형 잡힌 모델로 평가된 Random Forest에 대해 SHAP (SHapley Additive exPlanations) 분석을 수행하여 모델의 예측 과정을 투명하게 해석했습니다.
### 4.1. SHAP Summary Plot: 전역적 특징 중요도 분석
* **분석 결과**: concave points_worst, perimeter_worst, radius_worst 순으로 악성($M=1$) 예측에 가장 큰 기여를 했습니다.
* **특징 해석**: 플롯에서 **붉은색 점(특징의 높은 값)**이 **양의 $\text{SHAP}$ 값(악성 예측 증가)**으로 광범위하게 분포하는 것을 확인했습니다.
* **임상적 의미**: 종양의 최종 상태(worst), 특히 **경계의 불규칙성(concave points)**과 **크기(perimeter, radius)**가 클수록 악성일 확률이 강하게 증가합니다.
* <img width="771" height="974" alt="shap_summary_plot" src="https://github.com/user-attachments/assets/3f140665-7faa-433e-a27d-b67f09b23ce5" />



## 4.2. SHAP Dependence Plot: 비선형적 관계 및 상호작용 분석
concave points_worst 특징에 대한 SHAP의존성 플롯을 분석하여 모델이 학습한 복잡한 패턴을 시각화했습니다.
<img width="646" height="488" alt="shap_dependence_plot" src="https://github.com/user-attachments/assets/c4cd8099-43ff-43c6-93c6-8262bdfb00bf" />
X축의 concave points_worst 값이 특정 지점 이상으로 증가할 때, Y축의 SHAP 값(악성 예측 기여도)이 급격하게(비선형적) 상승하는 패턴을 보입니다.
* **비선형성 해석**: 이는 선형 모델로는 포착하기 어려운, **"특징 값이 임계치를 넘어설 때 악성 위험이 기하급수적으로 증가한다"**는 임상적으로 중요한 복잡한 관계를 $\text{Random Forest}$가 효과적으로 학습했음을 의미합니다.
* **상호작용 효과**: 플롯 내의 수직적 색상 변화는 concave points_worst 외의 다른 특징과 예측 간에 상호작용 효과가 존재함을 강력하게 암시합니다.

## ⚙️ 5. 특징 중요도 분석 (Feature Importance)

```python
# --- SVM, MLP 모델: Permutation Importance ---
    
    perm_importance_svm = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    sorted_idx_svm = perm_importance_svm.importances_mean.argsort()

    perm_importance_mlp = permutation_importance(mlp_model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    sorted_idx_mlp = perm_importance_mlp.importances_mean.argsort()

    # --- Random Forest 모델: Gini Importance (Mean Decrease in Impurity) ---
    rf_importance = rf_model.feature_importances_
    sorted_idx_rf = rf_importance.argsort()
```
<img width="2400" height="1000" alt="feature_importance" src="https://github.com/user-attachments/assets/49227e71-c6c0-4c3c-9cc3-d3689051406c" />

## ⚙️ 5.1 분석 결과

* 1. 특징 중요도 분석 결과, 세 모델 모두에서 'worst' 접두사가 붙은 특징들이 진단에 핵심적인 역할을 함을 확인했습니다.
➡️종양의 최종 상태가 악성/양성 판별에 가장 결정적인 정보임을 시사합니다.
*2.
  *  세 모델 중 SVM 가장 높은 AUC를 기록했으나 그 차이는 미미합니다.
  *  Random Forest는 높은 성능과 함께 자체적인 특징 중요도 정보를 제공하는 장점이 있습니다.
  *  MLP 은  매우 경쟁력 있는 성능을 보여주었습니다.
따라서 실제 임상 적용 시, 높은 성능을 유지하면서도 모델의 작동 방식을 설명하기 용이한 Random Forest나 SVM이 복잡한 MLP보다 더 선호될 수 있습니다.
```python
# --- 각 모델의 예측 확률 계산 ---
    y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    # --- ROC 커브 및 AUC 계산 ---
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    print(f"\nSVM 모델 AUC: {roc_auc_svm:.4f}")
    print(f"MLP 모델 AUC: {roc_auc_mlp:.4f}")
    print(f"Random Forest 모델 AUC: {roc_auc_rf:.4f}")
```
<img width="1000" height="800" alt="roc_curve_comparison" src="https://github.com/user-attachments/assets/54e10ba7-e245-4cbe-85dd-6553169a10a3" />


## 📜 6. 결론 및 제언 (Conclusion & Suggestion)

### 최종요약
* **성능**: 모든 모델이 목표 AUC 를 초과 달성했으며,SVM이 0.9960으로 가장 높은 {Test AUC}를 기록했습니다.
* **신뢰성**: k-fold cv를 통해 svm과 Random Forest 모두 매우 안정적인 성능을 보였습니다.
* **해석력**:SHAP 분석을 통해 **worst concave points**와 **worst radius**가 악성 진단의 핵심 요인이며, 이들이 예측에 비선형적으로 기여함을 명확히 해석했습니다.
  
### 최종 모델 성능 (ROC-AUC)
| 모델 | 최종 AUC 점수 (테스트 세트) |
| :--- | :--- |
| **SVM** | **0.9960** |
| **MLP** | **0.9950** |
| **Random Forest** | **0.9929** |

