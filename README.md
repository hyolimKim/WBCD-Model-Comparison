# WBCD-Model-Comparison
# 🎀 유방암 진단 유효성 비교 및 특징 중요도 해석 연구 (WBCD 기반)

## ⭐️ 1. 프로젝트 개요 (Project Overview)

### 🧐 문제정의(Problem Identification)
유방암은 조기 발견 시 생존율이 증가함에도 불구하고, 기존 세포병리 기반의 진단은 검사자의 경험, 판독의 주관성, 그리고 **특징 변수 간의 복잡한 비선형 관계**로 인해 오진 가능성이 존재합니다. 따라서 실제 임상에서는 악성/양성 판단을 보다 정량적인 방법으로 보완할 필요성이 지속적으로 제기되고 있습니다.

본 연구에서는 유방암 진단의 정량적 의사결정 지원 시스템 구축을 위해, 세포 병리학적 특징을 활용해 **악성/양성 여부를 예측하는 머신러닝 모델(SVM, MLP, Random Forest)** 을 비교합니다.
특히 임상적 활용 가능성을 높이기 위해

* 어떤 모델이 **가장 높은 진단 정확도(AUC)** 를 보이는지 비교하고,
* 각 모델이 어떤 **세포 형태학적 특징(feature)** 을 가장 중요하게 활용하는지 해석하며,
* 이 특징들이 실제 **임상적 의미**와 어떻게 연결되는지를 분석합니다.

이를 통해 본 연구는
▶ **병리 판독의 보조 도구** 로 활용 가능한 모델을 탐색하고,
▶ 임상의가 수용할 수 있는 **설명력** 을 갖춘 모델의 필요성을 해결하는 것을 목표로 합니다.

### 🎯 프로젝트 목표 (Project Goal)
1. **모델 유효성 비교**: **SVM, MLP, Random Forest ** 모델의 유방암 진단 성능을 ROC-AUC를 기준으로 비교 분석합니다.
2. **특징 중요도 해석**: 모델별 특징 중요도를 산출하여 진단에 영향을 미치는 세포 형태학적 특징(feature)을 식별하고 임상적 의미를 해석합니다.
3. **최적 모델 제시**: 통계적 유효성과 해석 가능성을 고려하여 임상 환경에 더 적합한 모델을 제안합니다.
---
### 🎯 목표모델 성능 수준 (Target Performance)
* **SVM 모델**: $AUC \ge 0.96$
* **MLP 모델**: $AUC \ge 0.97$
* **rf 모델**: $AUC \ge 0.95$

## 💾 2. 데이터 소개 및 탐색 (Data Introduction & EDA)

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
* Permutation Importance는 특정 특징의 값을 무작위로 섞었을 때 모델 성능이 얼마나 감소하는지를 측정합니다.
* 모델 종류에 상관없이 적용 가능하며, 다중공선성이 있는 데이터에서도 신뢰도가 높습니다.

`radius`, `perimeter`, `area` 등 세포핵의 크기와 관련된 특징들 간에 **다중공선성**이 관찰되었습니다. 
하지만 해석력의 상실을 방지하기 위해  PCA를 적용하지 않고, 상관관계가 높다는 사실을 인지한 상태에서 **Random Forest** 같은 트리 기반 모델과, 
**Permutation Importance**로 각 변수의 순수한 영향력을 분석합니다.

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
<img width="1000" height="800" alt="roc_curve_comparison" src="https://github.com/user-attachments/assets/54e10ba7-e245-4cbe-85dd-6553169a10a3" />


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

## ⚙️ 6. 분석 결과

* 1. 특징 중요도 분석 결과, 세 모델 모두에서 'worst' 접두사가 붙은 특징들이 진단에 핵심적인 역할을 함을 확인했습니다.
➡️종양의 최종 상태가 악성/양성 판별에 가장 결정적인 정보임을 시사합니다.
*2.
  *  세 모델 중 SVM 가장 높은 AUC를 기록했으나 그 차이는 미미합니다.
  *  Random Forest는 높은 성능과 함께 자체적인 특징 중요도 정보를 제공하는 장점이 있습니다.
  *  MLP 은  매우 경쟁력 있는 성능을 보여주었습니다.
따라서 실제 임상 적용 시, 높은 성능을 유지하면서도 모델의 작동 방식을 설명하기 용이한 Random Forest나 SVM이 복잡한 MLP보다 더 선호될 수 있습니다.
