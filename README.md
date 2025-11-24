# WBCD-Model-Comparison
# 🎀 유방암 진단 유효성 비교 및 특징 중요도 해석 연구 (WBCD 기반)

## ⭐️ 1. 프로젝트 개요 (Project Overview)

### 🧐 문제정의(Problem Identification)
유방암 악성/양성 분류 문제에서 **SVM 모델**,**MLP 모델** ,**rf모델** 중 어떤 모델이 더 높은 진단 유효성(ROC-AUC)을 가지는지 정량적으로 비교하고, 각 모델의 **주요 특징(Feature)**을 해석합니다.

### 🎯 프로젝트 목표
1. **모델 유효성 비교**: **SVM, MLP, Random Forest ** 모델의 유방암 진단 성능을 ROC-AUC를 기준으로 비교 분석합니다.
2. **특징 중요도 해석**: 모델별 특징 중요도를 산출하여 진단에 영향을 미치는 핵심 의료 특징(feature)을 식별하고 임상적 의미를 해석합니다.
3. **최적 모델 제시**: 통계적 유효성과 해석 가능성을 고려하여 임상 환경에 더 적합한 모델을 제안합니다.
---
### 🎯 목표모델 성능 수준 ((Target Performance)
* **SVM 모델 (베이스라인)**: $AUC \ge 0.96$
* **MLP 모델 (비교 모델)**: $AUC \ge 0.97$
* **rf 모델 (비교 모델)**: $AUC \ge 0.95$

### 🎯 타깃 환자
* 위스콘신 대학병원에서 유방암 진단을 받은 가상의 환자 코호트 (세포 검사 기반의 악성/양성 분류).

## 💾 2. 데이터 소개 및 탐색 (Data Introduction & EDA)

### 📌 분석 대상 데이터 (WBCD)
* **데이터 출처:** UCI Machine Learning Repository (Wisconsin Breast Cancer Diagnostic Dataset)
* **분석 대상:** 미세침 흡인(FNA) 이미지에서 추출된 세포 핵의 형태적 특징
* **변수 구성:** 총 32개
    * **ID:** 환자 식별 번호 (제거)
    * **Diagnosis (타겟):** **M (악성, 1)** 또는 **B (양성, 0)**
    * **30개 Feature:** 10가지 세포핵 측정 항목 (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)에 대해 **Mean, Standard Error (SE), Worst** 통계량을 계산하여 구성.
* **샘플 개수:** 총 569개 (양성 357개, 악성 212개)
* **결측치:** 없음
 

### 📊 주요 데이터 탐색 결과 (EDA)
#### 1. 타겟 변수 분포
악성(1)과 양성(0) 샘플 비율이 약 2:3으로 **약간의 불균형**은 있으나, 모델 학습을 방해할 정도는 아닙니다. `stratify` 옵션을 사용하여 학습 및 테스트 세트에서 이 비율을 유지했습니다.

#### 2. 특징 간 상관관계 (Mean Features)
`radius`, `perimeter`, `area` 등 세포핵의 크기와 관련된 특징들 간에 **매우 높은 양의 상관관계(다중공선성)**가 관찰되었습니다. 이는 모델의 해석(예: 회귀 계수)을 어렵게 만들 수 있어, 본 분석에서는 **Permutation Importance**와 같은 더 신뢰성 있는 해석 기법을 사용했습니다.

---

## ⚙️ 3. 모델 학습 및 전처리 코드 요약

### 1. 전처리 (Preprocessing)
불필요한 열을 제거하고, 타겟 변수를 숫자로 변환한 후, **StandardScaler**를 이용하여 모든 특징을 표준화했습니다. 거리에 민감한 SVM, MLP 모델의 안정적인 학습을 위해 필수적인 과정입니다.

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

###2. SVM 하이퍼파라미터 튜닝
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
## ⚙️ 🏆 4. 목표 프로젝트 결과물 및 성능 비교 (수치 업데이트)

### 최종 모델 성능 (ROC-AUC)
| 모델 | 최종 AUC 점수 (테스트 세트) |
| :--- | :--- |
| **SVM** | **0.9960** |
| **MLP** | **0.9950** |
| **Random Forest** | **0.9929** |

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
## ⚙️ 5. 특징 중요도 분석 (Feature Importance)

```python
# --- SVM, MLP 모델: Permutation Importance ---
    # Permutation Importance는 특정 특징의 값을 무작위로 섞었을 때 모델 성능이 얼마나 감소하는지를 측정합니다.
    # 모델 종류에 상관없이 적용 가능하며, 다중공선성이 있는 데이터에서도 신뢰도가 높습니다.
    perm_importance_svm = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    sorted_idx_svm = perm_importance_svm.importances_mean.argsort()

    perm_importance_mlp = permutation_importance(mlp_model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    sorted_idx_mlp = perm_importance_mlp.importances_mean.argsort()

    # --- Random Forest 모델: Gini Importance (Mean Decrease in Impurity) ---
    # Random Forest는 모델 훈련 과정에서 각 특징이 불순도(impurity)를 얼마나 감소시키는지를 기반으로 중요도를 계산합니다.
    # 계산 속도가 빠르지만, 상관관계가 높은 특징들 사이에서는 중요도가 한쪽으로 쏠릴 수 있습니다.
    rf_importance = rf_model.feature_importances_
    sorted_idx_rf = rf_importance.argsort()
```
<img width="2400" height="1000" alt="feature_importance" src="https://github.com/user-attachments/assets/49227e71-c6c0-4c3c-9cc3-d3689051406c" />
