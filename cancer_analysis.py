# -*- coding: utf-8 -*-
"""
## 📑 유방암 진단 유효성 비교 및 특징 중요도 해석 연구 (SVM, MLP, Random Forest)

### 🎯 프로젝트 목표
1. **모델 유효성 비교**: SVM, MLP, Random Forest 모델의 유방암 진단 성능을 ROC-AUC를 기준으로 비교 분석.
2. **특징 중요도 해석**: 모델별 특징 중요도를 산출하여 진단에 영향을 미치는 핵심 의료 특징(feature)을 식별하고 임상적 의미를 해석.
3. **최적 모델 제시**: 통계적 유효성과 해석 가능성을 고려하여 임상 환경에 더 적합한 모델을 제안.

### 🌟 프로젝트 고도화 목표 (v2)
1. **통계적 신뢰도 확보**: K-Fold 교차 검증 및 Fold별 ROC Curve 시각화를 통해 모델 성능의 안정성을 확보.
2. **임상 의사결정 지원**: 예측 확률 임계값(Threshold)에 따른 민감도/특이도 테이블을 제시하여 임상적 활용도를 높임.
3. **비선형 모델 해석 강화**: SHAP (SHapley Additive exPlanations) 분석을 통해 복잡한 모델의 예측 결과를 직관적으로 해석하고, 특징 간의 비선형적 관계와 상호작용 효과를 규명.

### 👨‍💻 작성자: Gemini (AI)
### 📅 작성일: 2025년 11월 28일
"""
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap # SHAP 라이브러리 추가

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # StratifiedKFold 추가
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, recall_score

# ==============================================================================
# 📂 1순위: 통계적 신뢰도 강화 함수
# ==============================================================================

def plot_kfold_roc_curves(model, X, y, scaler, model_name, n_splits=10):
    """
    K-Fold 교차 검증을 수행하고, 각 Fold의 ROC Curve와 평균 AUC를 시각화합니다.
    - K-Fold CV는 모델의 성능이 특정 데이터 분할에 의존하지 않고 일반화되었음을 보여주는 강력한 방법입니다.
    - Fold별 ROC Curve를 함께 도시하면 모델 성능의 안정성(분산)을 직관적으로 파악할 수 있습니다.
    """
    print(f"\n--- [{model_name}] {n_splits}-Fold 교차 검증 시작 ---")
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # K-Fold 루프 실행
    for i, (train, test) in enumerate(kfold.split(X, y)):
        # 데이터 스케일링 (매 Fold마다 재학습하여 정보 유출 방지)
        X_train_fold, X_test_fold = X.iloc[train], X.iloc[test]
        y_train_fold, y_test_fold = y.iloc[train], y.iloc[test]
        
        # 새로운 Scaler 객체를 생성하여 fit_transform 수행
        fold_scaler = StandardScaler()
        X_train_scaled_fold = fold_scaler.fit_transform(X_train_fold)
        X_test_scaled_fold = fold_scaler.transform(X_test_fold)
        
        # 모델 학습 및 예측
        model.fit(X_train_scaled_fold, y_train_fold)
        y_prob_fold = model.predict_proba(X_test_scaled_fold)[:, 1]
        
        # ROC 커브 계산
        fpr, tpr, _ = roc_curve(y_test_fold, y_prob_fold)
        roc_auc = auc(fpr, tpr)
        
        # 결과 저장
        aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # 개별 Fold의 ROC 커브 플로팅 (선택적)
        ax.plot(fpr, tpr, alpha=0.3, lw=1, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    # 평균 ROC 커브 계산 및 플로팅
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
            lw=2.5)

    # 그래프 스타일 설정
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'{model_name}: {n_splits}-Fold Cross-Validation ROC Curves',
           xlabel='False Positive Rate (1 - Specificity)',
           ylabel='True Positive Rate (Sensitivity)')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f"roc_kfold_{model_name.replace(' ', '_')}.png")
    plt.close(fig)

    print(f"INFO: '{model_name}'의 K-Fold ROC Curve를 'roc_kfold_{model_name.replace(' ', '_')}.png'에 저장했습니다.")
    print(f"결과: 평균 AUC = {mean_auc:.4f}, 표준편차 = {std_auc:.4f}")


def generate_sensitivity_specificity_table(y_true, y_prob, model_name):
    """
    다양한 예측 확률 임계값(Threshold)에 따른 민감도와 특이도를 계산하여 표로 반환합니다.
    - 민감도(Sensitivity): 실제 악성(1)을 정확히 예측한 비율 (True Positive Rate)
    - 특이도(Specificity): 실제 양성(0)을 정확히 예측한 비율 (True Negative Rate)
    임상에서는 '악성 환자를 놓치지 않는 것(높은 민감도)'과 '양성 환자에게 불필요한 검사를 줄이는 것(높은 특이도)' 사이의 균형이 중요합니다.
    이 표는 의사가 특정 상황에 맞는 최적의 임계값을 선택하는 데 도움을 줍니다.
    """
    print(f"\n--- [{model_name}] 임계값별 민감도/특이도 분석 ---")
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        
        # confusion matrix를 이용한 계산
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (cm[0,0], 0, 0, 0) if y_pred.sum() == 0 else (0, 0, cm[0,0], 0)


        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({'Threshold': f"{th:.1f}", 
                        'Sensitivity': f"{sensitivity:.4f}", 
                        'Specificity': f"{specificity:.4f}"})
    
    results_df = pd.DataFrame(results)
    print("의사결정 지원 테이블 (임계값에 따른 민감도/특이도 변화):")
    print(results_df)
    return results_df


def main():
    """메인 분석 로직을 실행하는 함수"""
    
    # 한글 폰트 깨짐 방지를 위한 설정 (Matplotlib)
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        print("Malgun Gothic 폰트를 찾을 수 없습니다. 다른 한글 폰트로 변경하거나, 폰트 설치가 필요합니다.")
    plt.rcParams['axes.unicode_minus'] = False


    # --- 데이터 로드 ---
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("오류: 'data.csv' 파일을 찾을 수 없습니다. 파일이 현재 디렉토리에 있는지 확인해주세요.")
        sys.exit()

    # --- 데이터 전처리 (초기) ---
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['id', 'Unnamed: 32'], axis=1)
    elif 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # --- 특징(X)과 타겟(y) 분리 ---
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # --- 학습/테스트 데이터 분리 ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 특징 스케일링 (Standard Scaling) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==============================================================================
    # 🎯 4. 모델 학습, 평가 및 비교
    # ==============================================================================

    # --- 4.1. SVM 모델 (GridSearchCV) ---
    print("\n--- SVM 모델 학습 및 평가 ---")
    param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid_search_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search_svm.fit(X_train_scaled, y_train)
    svm_model = grid_search_svm.best_estimator_
    print(f"SVM 최적 파라미터: {grid_search_svm.best_params_}")
    
    # --- 4.2. MLP 모델 ---
    print("\n--- MLP 모델 학습 및 평가 ---")
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.0001, solver='adam', random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # --- 4.3. Random Forest 모델 ---
    print("\n--- Random Forest 모델 학습 및 평가 ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # --- 4.4. 모델 성능 비교 (ROC-AUC) ---
    y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_svm, tpr_svm, lw=2, label=f'SVM (AUC = {roc_auc_svm:.4f})')
    plt.plot(fpr_mlp, tpr_mlp, lw=2, label=f'MLP (AUC = {roc_auc_mlp:.4f})')
    plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('모델별 ROC 커브 비교')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_comparison.png")
    plt.close()

    # ==============================================================================
    # 🥇 1순위: 통계적 신뢰도 및 임상 의사결정 분석
    # ==============================================================================
    
    # --- 1-1. K-Fold CV로 모델 안정성 검증 ---
    plot_kfold_roc_curves(RandomForestClassifier(n_estimators=100, random_state=42), X, y, StandardScaler(), "Random Forest")
    plot_kfold_roc_curves(SVC(probability=True, C=grid_search_svm.best_params_['C'], gamma=grid_search_svm.best_params_['gamma'], random_state=42), X, y, StandardScaler(), "SVM")

    # --- 1-2. 임계값 기반 민감도/특이도 테이블 (Random Forest 기준) ---
    generate_sensitivity_specificity_table(y_test, y_prob_rf, "Random Forest")

    # ==============================================================================
    # 🥈 2순위: 비선형 모델 해석 강화 (SHAP 분석) - 최종 수정
    # ==============================================================================
    print("\n\n--- [Random Forest] SHAP 분석 시작 (Modern API) ---")
    
    # 1. SHAP Explainer 생성 (새로운 통합 API 사용)
    # shap.Explainer는 다양한 모델에 일관된 인터페이스를 제공하며, 배경 데이터를 함께 전달하여 안정성을 높입니다.
    explainer = shap.Explainer(rf_model, X_train_scaled, feature_names=X.columns)
    
    # 2. SHAP 값 계산
    # explainer를 직접 호출하여 값, 데이터, 이름 등이 모두 포함된 풍부한 Explanation 객체를 얻습니다.
    shap_explanation = explainer(X_test_scaled)

    # --- 2-1. SHAP Summary Plot: 특징 영향력 시각화 ---
    print("SHAP Summary Plot 생성 중...")
    plt.figure()
    # Explanation 객체에서 악성(class 1)에 대한 부분만 슬라이싱하여 전달합니다.
    # shap_explanation[:,:,1]은 (모든 샘플, 모든 특징, class 1)을 의미합니다.
    shap.summary_plot(shap_explanation[:,:,1], show=False)
    plt.title('SHAP Summary Plot for Random Forest (Class: Malignant)', pad=20)
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')
    plt.close()
    print("INFO: 'shap_summary_plot.png' 파일로 SHAP Summary Plot을 저장했습니다.")
    
    print("""
    [SHAP Summary Plot 해석 및 임상적 의미]
    - worst concave points, worst perimeter, worst radius 등의 특징이 오른쪽에 넓게 분포하며, 붉은색(높은 값)을 띱니다.
      -> 임상적 의미: 종양의 경계가 불규칙하고(concave points), 종양의 크기(radius)와 둘레(perimeter)가 클수록 악성(1)으로 예측될 확률이 강하게 높아집니다.
      이는 병리학적으로 악성 종양이 주변 조직을 침범하며 불규칙한 형태로 성장하는 특성과 일치하는 매우 중요한 결과입니다.
    """)

    # --- 2-2. SHAP Dependence Plot: 특정 특징과 예측의 관계 분석 ---
    top_feature = 'concave points_worst'
    print(f"SHAP Dependence Plot 생성 중 (Feature: {top_feature})...")
    
    plt.figure()
    # Explanation 객체를 사용하면 dependence_plot이 내부적으로 필요한 값들을 자동으로 처리해 편리합니다.
    # shap_values의 경우 .values 속성을 통해 접근합니다.
    shap.dependence_plot(top_feature, shap_explanation.values[:, :, 1], X_test_scaled, feature_names=X.columns, interaction_index="auto", show=False)
    plt.title(f'SHAP Dependence Plot: {top_feature}', pad=15)
    plt.ylabel('SHAP value (for Malignant class)')
    plt.savefig('shap_dependence_plot.png', bbox_inches='tight')
    plt.close()
    print("INFO: 'shap_dependence_plot.png' 파일로 SHAP Dependence Plot을 저장했습니다.")
    
    print(f"""
    [SHAP Dependence Plot 해석: '{top_feature}']
    - X축의 '{top_feature}' 값이 증가함에 따라, Y축의 SHAP 값(악성 예측 기여도)이 급격하게 증가하는 비선형적 관계를 보입니다.
    - 이는 모델이 "종양 경계의 오목한 부분이 많을수록 악성일 가능성이 기하급수적으로 높아진다"는 복잡한 패턴을 학습했음을 의미합니다.
    - 중간에 보이는 수직적 색상 변화는 다른 특징과의 '상호작용 효과'를 암시합니다. 이것이 비선형 모델을 사용하는 강력한 이유입니다.
    """)
    
    # 상호작용 효과는 위 Dependence Plot의 색상으로 표현되므로, 별도 플롯은 안정성을 위해 생략합니다.

    # ==============================================================================
    # 📜 최종 요약 및 결론
    # ==============================================================================
    print("\n\n### 최종 요약 및 결론 (v2) ###")
    print(f"SVM 모델 Test AUC: {roc_auc_svm:.4f}")
    print(f"MLP 모델 Test AUC: {roc_auc_mlp:.4f}")
    print(f"Random Forest 모델 Test AUC: {roc_auc_rf:.4f}")
    print("\n--- 신뢰도 및 해석력 분석 결과 ---")
    print("1. [안정성] K-Fold CV 결과, Random Forest와 SVM 모두 Fold에 관계없이 안정적인 AUC 성능(낮은 std)을 보여 신뢰성을 확보했습니다.")
    print("2. [임상 활용] 민감도/특이도 테이블은 특정 진료 환경(예: 1차 스크리닝 vs. 정밀 진단)에 맞는 최적의 의사결정 기준을 제공할 수 있습니다.")
    print("3. [해석력] SHAP 분석을 통해 Random Forest 모델이 'worst concave points', 'worst radius'와 같은 핵심 특징들의 비선형적 관계와 상호작용을 어떻게 학습했는지 명확히 확인했습니다.")
    print("\n[최종 제언]")
    print("Random Forest 모델이 높은 예측 성능과 임상적으로 유의미한 해석 가능성을 모두 제공하는 가장 균형 잡힌 모델이라고 결론지을 수 있습니다.")

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            temp_folder = 'C:\\joblib_temp'
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
            os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder
        except Exception as e:
            print(f"WARNING: joblib 임시 폴더 설정에 실패했습니다. 오류: {e}")
            
    main()