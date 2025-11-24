# WBCD-Model-Comparison
WBCD 기반 유방암 악성/양성 진단 유효성 비교 및 특징 중요도 해석 연구
📑 유방암 진단 유효성 비교 및 특징 중요도 해석 연구 (SVM, MLP, Random Forest)

📌 1. 프로젝트 개요 (Project Overview)
🎯 프로젝트 목표
1. **모델 유효성 비교**: SVM, MLP, Random Forest 모델의 유방암 진단 성능을 ROC-AUC를 기준으로 비교 분석.
2. **특징 중요도 해석**: 모델별 특징 중요도를 산출하여 진단에 영향을 미치는 핵심 의료 특징(feature)을 식별하고 임상적 의미를 해석.
3. **최적 모델 제시**: 통계적 유효성과 해석 가능성을 고려하여 임상 환경에 더 적합한 모델을 제안.

본 프로젝트는 머신러닝 모델을 활용하여 유방암을 진단하는 분류기의 유효성을 비교하고, 각 진단 특징(Feature)이 결과에 미치는 중요도를 해석하는 데 중점을 둡니다.

배경: 유방암은 조기 진단이 매우 중요하며, 객관적인 세포핵 특징을 기반으로 악성/양성을 정확하게 분류하는 모델은 임상 진단 보조 도구로 활용될 잠재력이 높습니다.

핵심 목표: 높은 진단 정확도(AUC $\geq 0.96$)를 달성함과 동시에, 의료진이 모델의 판단 근거를 이해할 수 있도록 **해석 가능성(Interpretability)**이 높은 최적 모델을 제시합니다.

타깃 사용자:

1차: 유방암 진단 보조 시스템 구축에 관심 있는 연구자 및 데이터 분석가

2차: 유방암 진단에 필요한 세포핵 특징의 중요도를 객관적으로 확인하려는 의료 관계자

🔬 2. 데이터셋 소개 및 탐색 (Data & EDA)

2.1. 데이터 종류 및 구조

데이터셋: 위스콘신 유방암 진단 데이터셋 (Wisconsin Breast Cancer Diagnostic, WBCD)

샘플 수: 569개 ($\text{양성}: 357, \text{악성}: 212$)

특징 수: 30개 (세포핵 특징) + $\text{id}$, $\text{diagnosis}$

데이터 출처: UCI Machine Learning Repository (일반적으로 공개 의료 데이터셋)

데이터 필드

설명

예시

diagnosis (타겟)

진단 결과 ($\text{0}: \text{양성} (\text{Benign}), \text{1}: \text{악성} (\text{Malignant})$)

object → int로 변환하여 사용

Mean Features

종양 세포핵의 평균값 ($\text{radius, texture, perimeter, area}$ 등 10개)

radius_mean

SE Features

세포핵 특징의 표준 오차 ($\text{Standard Error}$)

texture_se

Worst Features

세포핵 특징 중 가장 심한/나쁜 값 (최대/평균)

perimeter_worst

2.2. 탐색적 데이터 분석 (EDA)

타겟 분포: 진단 결과(diagnosis)는 악성(212개)과 양성(357개)으로 구성되어 있으며, 비교적 심한 불균형 없이 모델 학습에 안정적인 분포를 보입니다. * 상관관계 분석: radius, perimeter, area 등 크기와 관련된 특징들 간에는 매우 높은 양의 상관관계가 나타났습니다. 이는 다중공선성(Multicollinearity) 문제를 시사하며, 모델의 계수 기반 해석보다 Permutation Importance와 같은 방법을 사용하는 것이 특징 중요도 분석의 신뢰도를 높입니다.

💻 3. 모델링 및 성능 비교 (Modeling & Comparison)

3.1. 모델 개요

진단 유효성 및 해석 가능성을 비교하기 위해 세 가지 주요 머신러닝 분류 모델을 선정했습니다.

모델

특징

장점

SVM (Support Vector Machine)

최적의 결정 경계(Hyperplane)를 찾아 분류

적은 데이터로도 높은 성능, 구조적 간결성

MLP (Multi-Layer Perceptron)

복잡한 비선형 관계 학습에 능숙한 심층 신경망

비선형 관계에서 가장 높은 잠재 성능

Random Forest

여러 결정 트리를 조합한 앙상블 모델

과적합 방지에 강하고, 자체적인 특징 중요도 제공

3.2. 데이터 전처리 및 학습 전략

데이터 분리: 전체 데이터셋을 학습(Train, 80%) 및 테스트(Test, 20%) 셋으로 분리하고, 타겟 변수 비율을 유지(stratify=y)하여 데이터의 재현성을 확보했습니다.

스케일링: $\text{SVM}$과 $\text{MLP}$의 안정적인 학습을 위해 StandardScaler를 사용하여 특징 변수들을 표준화했습니다.

3.3. 프로젝트 결과물 및 성능 지표

세 모델 모두 목표 성능 수준을 초과 달성하며 높은 진단 유효성을 입증했습니다.

모델

Accuracy

AUC (목표치)

F1-Score (악성, $\text{1}$)

SVM

$0.97$

$0.9947$ ($\geq 0.96$)

$0.96$

MLP

$0.98$

$0.9950$ ($\geq 0.97$)

$0.98$

Random Forest

$0.97$

$0.9929$ (참고)

$0.96$

ROC-AUC 분석: $\text{MLP}$ 모델이 $\text{AUC}$ **$0.9950$**으로 미세하게 가장 높은 성능을 기록했습니다. 이는 $\text{MLP}$가 데이터 내의 복잡한 비선형 패턴을 가장 잘 학습했음을 의미합니다. * 최종 평가: 세 모델 간의 성능 차이는 매우 미미하며, 모두 실제 임상 환경에서 사용 가능한 수준의 높은 유효성을 보여주었습니다.

📈 4. 특징 중요도 분석 및 임상적 해석

모델의 판단 근거를 확보하기 위해 각 모델별로 진단에 가장 큰 영향을 미치는 특징을 분석했습니다.

4.1. 특징 중요도 비교

SVM, MLP: $\text{Permutation Importance}$ (특징을 무작위로 섞었을 때 성능 감소 측정) 사용

Random Forest: $\text{Gini Importance}$ (불순도 감소량 측정) 사용

4.2. 핵심 해석

세 모델의 분석 결과는 일관된 결론을 제시합니다.

Worst Features의 중요성: 모든 모델에서 'worst' 접두사가 붙은 특징들(예: $\text{worst\_radius}$, $\text{worst\_perimeter}$, $\text{worst\_concave points}$)이 진단에 가장 핵심적인 역할을 수행했습니다.

임상적 의미: 이는 종양 세포핵이 가장 변형되고 악화된 상태(최종 상태)의 특징 정보가 악성/양성 판별에 가장 결정적인 정보임을 시사합니다. 평균(mean) 특징보다 종양의 최대/최악의 상태를 나타내는 특징이 진단 예측력이 높다는 결론을 얻을 수 있습니다.

4.3. 최적 모델 제안 (RWE 관점)

모델

성능 (AUC)

해석 가능성

임상 적용 적합성

MLP

최고 ($0.9950$)

낮음 (블랙박스)

복잡도가 높아 설명이 어려움

Random Forest

매우 높음 ($0.9929$)

중간 (자체 중요도 제공)

성능과 해석력이 균형 잡혀 선호됨

SVM

매우 높음 ($0.9947$)

중간/낮음

단순함에도 강력하여 경쟁력 높음

결론: 높은 성능을 유지하면서도 모델 작동 방식을 설명하기 용이한 Random Forest나 SVM이 복잡도가 높은 $\text{MLP}$보다 실제 임상 환경에서 설명 가능한 AI (XAI) 관점에서 더 선호될 수 있습니다.

⚙️ 5. 코드 분석 및 설명 (Code Analysis: cancer_analysis.py)

제출하신 파이썬 코드는 매우 체계적이며, 주석과 논리 구조가 훌륭합니다. 포트폴리오를 위해 각 섹션의 역할을 자세히 설명합니다.

A. 전처리 (Preprocessing)

코드 섹션

역할

학습 포인트

df.drop

불필요한 id 및 모든 값이 $\text{NaN}$인 Unnamed: 32 열 제거

데이터 클리닝의 기초이며, 모델 학습의 정확도를 높입니다.

df['diagnosis'].map

타겟 변수를 문자($\text{M}, \text{B}$)에서 숫자($\text{1}, \text{0}$)로 변환

머신러닝 모델은 숫자로 된 타겟 변수만 처리할 수 있습니다.

train_test_split

학습/테스트 데이터 분리 (test_size=0.2, $\text{20\%}$)

stratify=y 옵션을 통해 악성/양성 비율을 유지하여 편향되지 않은 테스트셋을 만듭니다.

StandardScaler

fit_transform 및 transform을 통한 데이터 스케일링

$\text{SVM}, \text{MLP}$ 등 거리 기반 모델의 성능 향상을 위한 필수 과정입니다. **학습 데이터에만 fit**하는 것은 데이터 유출을 방지하는 모범 사례입니다.

B. 모델 학습 및 평가

코드 섹션

역할

학습 포인트

SVC(..., probability=True)

$\text{SVM}$ 모델 정의 및 확률 예측 활성화

$\text{ROC-AUC}$ 계산을 위해서는 모델이 단순히 $\text{0/1}$ 예측이 아닌 확률 값을 출력해야 합니다.

MLPClassifier

은닉층 100개, 50개 뉴런을 가진 2층 $\text{MLP}$ 정의

은닉층의 개수와 뉴런 수는 모델의 복잡도를 결정하는 핵심 하이퍼파라미터입니다.

classification_report

$\text{Precision, Recall, F1-Score, Support}$ 등의 성능 지표 출력

단순히 $\text{Accuracy}$만 보는 것이 아니라, 악성(1) 예측에 대한 $\text{Precision}$과 $\text{Recall}$을 모두 확인하여 모델의 안정성을 평가합니다.

C. 최종 분석 (AUC & Feature Importance)

코드 섹션

역할

학습 포인트

roc_curve, auc

$\text{ROC}$ 곡선과 $\text{AUC}$ 점수를 계산 및 시각화

이진 분류 모델의 전역적인 유효성을 평가하는 가장 표준적인 지표입니다. 민감도($\text{TPR}$)와 특이도($\text{FPR}$)의 균형을 시각적으로 보여줍니다.

permutation_importance

$\text{SVM}$과 $\text{MLP}$의 특징 중요도 계산

$\text{Random Forest}$의 $\text{Gini Importance}$와 달리, 모델 학습 후에 독립적으로 계산되며, 다중공선성 문제를 피하는 가장 신뢰성 높은 XAI 기법 중 하나입니다.

한글 폰트 설정

$\text{Matplotlib}$ 그래프의 한글 깨짐 방지

plt.rcParams['font.family'] = 'Malgun Gothic' 등의 설정은 한국 환경에서 시각화 결과의 완성도를 높이는 실무적인 필수 코드입니다.
