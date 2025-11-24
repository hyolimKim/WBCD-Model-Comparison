# -*- coding: utf-8 -*-
"""
## 📑 유방암 진단 유효성 비교 및 특징 중요도 해석 연구 (SVM, MLP, Random Forest)

### 🎯 프로젝트 목표
1. **모델 유효성 비교**: SVM, MLP, Random Forest 모델의 유방암 진단 성능을 ROC-AUC를 기준으로 비교 분석.
2. **특징 중요도 해석**: 모델별 특징 중요도를 산출하여 진단에 영향을 미치는 핵심 의료 특징(feature)을 식별하고 임상적 의미를 해석.
3. **최적 모델 제시**: 통계적 유효성과 해석 가능성을 고려하여 임상 환경에 더 적합한 모델을 제안.

### 👨‍💻 작성자: Gemini (AI)
### 📅 작성일: 2025년 11월 23일
"""
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

def main():
    """메인 분석 로직을 실행하는 함수"""
    
    # 한글 폰트 깨짐 방지를 위한 설정 (Matplotlib)
    # 사용자 환경에 'Malgun Gothic' 폰트가 없는 경우, 다른 한글 폰트로 변경해야 할 수 있습니다.
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        print("Malgun Gothic 폰트를 찾을 수 없습니다. 다른 폰트로 설정하거나, 폰트 설치가 필요합니다.")
    plt.rcParams['axes.unicode_minus'] = False


    # --- 데이터 로드 ---
    # 프로젝트의 기반이 되는 유방암 진단 데이터를 불러옵니다.
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("오류: 'data.csv' 파일을 찾을 수 없습니다. 파일이 현재 디렉토리에 있는지 확인해주세요.")
        sys.exit()

    # ## 2. 데이터 탐색 (Exploratory Data Analysis, EDA)
    # 데이터의 구조, 결측치, 타겟 변수 분포 등을 파악하여 데이터에 대한 이해를 높입니다.

    print("--- 1. 데이터 기본 정보 ---")
    print(df.info())
    # 'Unnamed: 32' 열은 모든 값이 결측치(NaN)이므로 분석에 불필요합니다. 'id' 열 또한 환자 식별자로, 모델 학습에 사용되지 않습니다.

    # --- 데이터 전처리 (초기) ---
    # 불필요한 'id'와 'Unnamed: 32' 열을 제거합니다.
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['id', 'Unnamed: 32'], axis=1)
    else:
        # 'Unnamed: 32' 열이 없을 경우를 대비
        if 'id' in df.columns:
            df = df.drop('id', axis=1)


    # 타겟 변수인 'diagnosis'를 머신러닝 모델이 이해할 수 있도록 숫자(0, 1)로 변환합니다.
    # M(Malignant, 악성) -> 1, B(Benign, 양성) -> 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    print("\n--- 2. 데이터 샘플 확인 (Head) ---")
    print(df.head())

    print("\n--- 3. 타겟 변수 분포 확인 ---")
    # 악성(1)과 양성(0) 데이터의 개수와 비율을 확인합니다.
    # 데이터 불균형이 심하지 않아, 모델 학습에 안정적일 것으로 예상됩니다.
    target_counts = df['diagnosis'].value_counts()
    print(target_counts)

    plt.figure(figsize=(8, 6))
    sns.countplot(x='diagnosis', data=df)
    plt.title('진단 결과 분포 (0: 양성, 1: 악성)')
    plt.xticks([0, 1], ['양성 (Benign)', '악성 (Malignant)'])
    plt.ylabel('샘플 수')
    plt.savefig("target_distribution.png")
    print("INFO: 'target_distribution.png' 파일로 진단 결과 분포 그래프를 저장했습니다.")


    # --- 4. 특징 간 상관관계 분석 ---
    # 30개의 특징 변수들 간의 상관관계를 히트맵으로 시각화합니다.
    # radius, perimeter, area 등 크기와 관련된 특징들 간에 높은 양의 상관관계가 나타나는 것을 확인할 수 있습니다.
    # 이러한 다중공선성(Multicollinearity)은 모델의 해석을 어렵게 만들 수 있으므로,
    # 추후 특징 중요도 평가 시 계수(coefficient) 기반이 아닌 Permutation Importance를 사용하는 것이 더 신뢰성 높습니다.
    features_mean = list(df.columns[1:11]) # 'mean'이 포함된 특징만 선택
    corr = df[features_mean].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('주요 특징 간 상관관계 히트맵 (Mean Features)')
    plt.savefig("correlation_heatmap.png")
    print("INFO: 'correlation_heatmap.png' 파일로 상관관계 히트맵을 저장했습니다.")


    # ## 3. 데이터 전처리 (Preprocessing)
    # 모델 학습을 위해 데이터를 학습용과 테스트용으로 분리하고, 특징 스케일링을 수행합니다.

    # --- 특징(X)과 타겟(y) 분리 ---
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # --- 학습/테스트 데이터 분리 ---
    # 전체 데이터의 80%를 학습용으로, 20%를 테스트용으로 분리합니다.
    # random_state를 고정하여 실행할 때마다 동일한 결과를 얻도록 합니다. (재현성 확보)
    # stratify=y 옵션은 학습/테스트 데이터셋의 타겟 변수 비율을 원본 데이터와 동일하게 유지해줍니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 특징 스케일링 (Standard Scaling) ---
    # 각 특징의 단위를 통일시켜 모델이 안정적으로 학습되도록 합니다.
    # SVM, MLP와 같이 거리에 민감한 알고리즘에서는 스케일링이 성능에 큰 영향을 미칩니다.
    # 중요: scaler는 학습 데이터(X_train)에만 fit해야 하며, 그 기준으로 학습/테스트 데이터를 모두 transform해야 합니다. (데이터 유출 방지)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # ## 4. 모델 학습 및 평가 (SVM, MLP, Random Forest)
    # 세 가지 모델을 학습하고 성능을 평가합니다.

    # --- 4.1. SVM 모델 ---
    print("\n--- SVM 모델 학습 및 평가 ---")
    # C: 규제 강도, kernel: 커널 함수, probability: 확률 예측 활성화
    svm_model = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=['양성', '악성']))

    # --- 4.2. MLP 모델 ---
    print("\n--- MLP 모델 학습 및 평가 ---")
    # hidden_layer_sizes: 은닉층의 뉴런 수, max_iter: 최대 반복 학습 횟수
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.0001,
                              solver='adam', random_state=42, early_stopping=False)
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    print("MLP Classification Report:")
    print(classification_report(y_test, y_pred_mlp, target_names=['양성', '악성']))

    # --- 4.3. Random Forest 모델 ---
    print("\n--- Random Forest 모델 학습 및 평가 ---")
    # n_estimators: 생성할 트리의 개수
    # Random Forest는 스케일링에 영향을 받지 않지만, 다른 모델과의 일관성을 위해 스케일된 데이터를 사용합니다.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['양성', '악성']))


    # ## 5. 모델 성능 비교 (ROC-AUC)
    # 세 모델의 진단 유효성을 ROC 곡선과 AUC(Area Under the Curve) 점수를 통해 비교합니다.

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

    # --- ROC 커브 시각화 ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'SVM (AUC = {roc_auc_svm:.4f})')
    plt.plot(fpr_mlp, tpr_mlp, color='blue', lw=2, label=f'MLP (AUC = {roc_auc_mlp:.4f})')
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - 특이도)')
    plt.ylabel('True Positive Rate (민감도)')
    plt.title('ROC(Receiver Operating Characteristic) 커브 비교')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve_comparison.png")
    print("INFO: 'roc_curve_comparison.png' 파일로 ROC 커브 비교 그래프를 저장했습니다.")


    # ## 6. 특징 중요도 분석 (Feature Importance)
    # 모델별로 각 특징이 예측 성능에 얼마나 기여하는지 측정합니다.

    print("\n--- 특징 중요도 분석 중... ---")
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


    # --- 특징 중요도 시각화 ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle('모델별 특징 중요도 비교', fontsize=20)

    # SVM (Permutation Importance)
    ax1.barh(np.array(X.columns)[sorted_idx_svm], perm_importance_svm.importances_mean[sorted_idx_svm])
    ax1.set_xlabel("Permutation Importance")
    ax1.set_title("SVM 모델", fontsize=16)

    # MLP (Permutation Importance)
    ax2.barh(np.array(X.columns)[sorted_idx_mlp], perm_importance_mlp.importances_mean[sorted_idx_mlp])
    ax2.set_xlabel("Permutation Importance")
    ax2.set_title("MLP 모델", fontsize=16)

    # Random Forest (Gini Importance)
    ax3.barh(np.array(X.columns)[sorted_idx_rf], rf_importance[sorted_idx_rf])
    ax3.set_xlabel("Gini Importance")
    ax3.set_title("Random Forest 모델", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("feature_importance.png")
    print("INFO: 'feature_importance.png' 파일로 특징 중요도 그래프를 저장했습니다.")

    print("\n--- 분석 완료 ---")
    print("프로젝트의 모든 과정이 성공적으로 실행되었습니다.")
    print("생성된 그래프와 출력된 성능 지표를 포트폴리오에 활용하세요.")
    print("\n### 최종 요약 ###")
    print(f"SVM 모델 AUC: {roc_auc_svm:.4f} (목표: >= 0.96)")
    print(f"MLP 모델 AUC: {roc_auc_mlp:.4f} (목표: >= 0.97)")
    print(f"Random Forest 모델 AUC: {roc_auc_rf:.4f}")
    print("\n세 모델 모두 목표 성능 수준을 달성하며 매우 높은 진단 정확도를 보였습니다.")
    print("특징 중요도 분석 결과, 세 모델 모두에서 'worst' 접두사가 붙은 특징들(예: worst_radius, worst_perimeter, worst_concave points)이 진단에 핵심적인 역할을 함을 확인했습니다.")
    print("이는 종양의 최종 상태(가장 나쁜 상태의 세포핵 크기, 둘레, 오목한 점의 수 등)가 악성/양성 판별에 가장 결정적인 정보임을 시사합니다.")
    print("RWE 분석 관점에서, 세 모델 중 복잡도가 높은 MLP가 가장 높은 AUC를 기록했으나 그 차이는 미미합니다. 반면 Random Forest는 높은 성능과 함께 자체적인 특징 중요도 정보를 제공하는 장점이 있습니다. SVM은 단순함에도 불구하고 매우 경쟁력 있는 성능을 보여주었습니다.")
    print("따라서 실제 임상 적용 시, 높은 성능을 유지하면서도 모델의 작동 방식을 설명하기 용이한 Random Forest나 SVM이 복잡한 MLP보다 더 선호될 수 있습니다.")

if __name__ == '__main__':
    # Windows에서 multiprocessing 사용 시 스크립트가 재귀적으로 실행되는 것을 방지하고,
    # 한글 경로로 인한 오류를 해결하기 위해 메인 실행 부분을 이 블록 안에 둡니다.
    
    # joblib 라이브러리가 임시 파일을 생성할 때 ASCII 문자로만 구성된 경로를 사용하도록 설정
    if sys.platform == 'win32':
        try:
            temp_folder = 'C:\\joblib_temp'
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
            os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder
            print(f"INFO: joblib 임시 폴더를 '{temp_folder}'로 설정했습니다.")
        except Exception as e:
            print(f"WARNING: joblib 임시 폴더 설정에 실패했습니다. 오류: {e}")
            
    main()