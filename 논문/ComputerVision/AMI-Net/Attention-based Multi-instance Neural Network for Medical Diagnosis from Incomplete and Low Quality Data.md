# Attention-based Multi-instance Neural Network for Medical Diagnosis from Incomplete and Low Quality Data

## Abstract

임상 기록에서 패턴을 추출하는 한 가지 방법은 각 환자 기록을 증상의 형태로 다양한 수의 사례가 있는 bag으로 간주하는 것이다. 의학 진단은 유익한 것을 먼저 발견하고 나서 하나 이상의 질병에 매핑하는 것이다.

많은 경우 환자는 일부 feature space에서 vetors로 표현되며 진단 결과를 생성하기 위해 Classiier가 적용된다.

이 논문에서는 실제 환자 기록의 기존 정보와 유요한 정보만을 기반으로 하나의 질병을 classification하는 새로운 접근법인 AMI-Net을 제안한다.

환자의 경우, 인스턴스들의 한 bag을 입력으로 받아 end-to-end way로 직접 bag label을 출력한다.

인스턴스 간의 상관관계와 최종 분류에 대한 중요성은 multi-head attention transformer, instance-level multi-instance pooling 및 bag-level multi-instance pooling에 의해 파악된다.

제안된 접근 방식은 두 개의 표준화되지 않은 매우 imbalanced한 데이터 세트, 즉 한 개는 Traditional Chinese Medicine (TCM) domain과 다른 한 개는 WM(Western Medicine) 영역에서 테스트되었다.

## INTRODUCTION

Real-world 데이터는 (i) 데이터 정확도, (ii) 데이터 완전성, (iii) 데이터 일관성 및 (iv) 데이터 밸런스 같은 데이터 품질 문제의 대상이 된다.

input sample을 `a bag of instances` 라 하고, bag label만 주어진다.

Learning과 Training 동안, MIL models은 instances를 포함하는 새로운 bags의 labels를 예측한다.

본 논문에서, 우리는 주로 단일 질병 진단, 즉 단일 작업에 대한 이진 분류를 위한 MIL의 적용을 강조한다.

 MIL의 정의에 따라, bag은 적어도 한 가지 이상이 positive일 때만 positive으로 표시되고, 그렇지 않을 경우 가방은 negative으로 표시된다.

이러한 맥락에서 인스턴스 간의 상관 관계를 포착하고 가장 유용한 인스턴스를 찾는 것이 중요한 역할을 한다. 경우에 따라 인스턴스 간의 상관 관계를 무시하면 예측이 잘못될 수 있다.

의료 진단과 같은 복잡한 경과를 고려할 때 임상의가 독립적으로 위험요소를 평가할 뿐만 아니라 동시 발생의 영향도 고려해야 한다.

이것은 모델 구축 프로세스에서 인스턴스 상관 관계를 측정하는 것을 목표로 하는 출발점 중 하나이다.

인스턴스 간 또는 인스턴스와 bag 간의 관계를 포착하는 능력과 관련하여 attention 메커니즘은 몇 가지 성능 이점을 보여준다.

현재 이미지 및 텍스트 분석[20, 21]에서 두 가지 하위 범주로 널리 사용되고 있다: (i) task-supervised attention, (ii) self-supervised attention.

전자는 source와 target 사이의 관계를 포착하고, 후자는 source의 내부 관계(intra-relationship)를 계산(capture)한다. 두 sub-categories는 MIL에 필수적이다.

  이 연구에서, 우리는 그것들을 다중 인스턴스 신경망에 통합한다.

우리는 다음과 같이 불완전하고 품질이 낮은 데이터를 통해 의료 진단의 주요 작업에 접근한다:

(i) input instances를 embedding space에 매핑하고, 다른 embedding dimensions에 대해 서로 correlating하는 인스턴스를 매핑하여 일부 body condition을 나타낸다.

(ii) 서로 다른 임베딩 subspace에서 인스턴스 상관 관계를 캡처한다.

(iii) bag embedding을 학습하고, 

(iv) bag score를 얻기 위해 attention mechanism을 통해 informative instances를 선택한다.

모든 모듈은 MIL 신경망을 통해 매개 변수화되어 구조를 유연하고 단순하게 만든다.

이 접근 방식은 일부러 데이터를 수집하거나 수동으로 데이터를 선별할 필요가 없지만 자동으로 처리하여 최종 의료 진단을 지원하기 위해 많은 양의 저품질 데이터 중 가장 유용한 정보를 캡처한다.

## METHODOLOGY

전체적인 아키텍처는 embedding layer, residual connection이 있는 multi- head attention transformer, instance-wise fully connected layers set,instance-level MIL pooling layer 및 bag-level MIL pooling layer와 sigmoid function으로 구성된다.

![Screen Shot 2021-12-01 at 2.06.36 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-12-01 at 2.06.36 PM.png)

### Multi-instance Learning

Supervised learning function : `X -> Y`
$$
x_i \in X, Y_i \in Y
$$
MIL에서는, input dataset {X1,X2, ..., Xm}을 상응하는 label set {Y1,Y2,...Ym} 에 매핑하는 function (Yi ∈ {0,1})
$$
X_i : \text{a bag with a set of instances} \{ x_{i1}, x_{i2}, ... , x_{i,n_i}\} \\
n_i : \text{number of instances in } X_i
$$
각 Bag Xi에 대해 predict할 때, 적어도 하나의 instance label이 positive라면 bag level label은 positive가 된다. 
$$
Y_i=
\begin{cases}
0, & \mbox{all }y_{ij} = 0 \\
1, & \mbox{ otherwise}
\end{cases} \\
y_{ij} : \mbox{label of j-th instance in the i-th bag}
$$
위의 가정은 MIL의 기초, permutation invariance property, 그리고 MIL 문제를 해결하기 위한 permutation invariant symmetric function는 다음과 같은 함수로 나타낼 수 있다.
$$
f(X) = \theta(\eta_{x\in X}\varphi(x)) \\

\varphi, \theta : \mbox{suitable transformations} \\ 
\eta : \mbox{permutation invariance function (ex : MIL pooling)} \\
\theta : \mbox{scoring function for a bag of instances}
$$
적합한 위 파리미터를 선택하는 다양한 선택과 관련하여, 두 가지 주요 MIL 접근법이 있다.

(i) Instance-level MIL pooling approach : varphi는 instance transformer이고 MIL pooling function인 eta는 각 instance에서 채택되어 bag classifier theta에 의한 추가 procession을 위해 bag embedding을 얻는다.

(ii) Bag-level MIL pooling approach : varphi는 instance score를 얻기 위한 transformation이며, 이 transformation은 bag socre를 얻기 위해 MIL Pooling eta에 의해 추가로 처리되고, theta는 injective function(일대일 함수)이다.

**MIL with Neural Networks**

위의 MIL 기반 함수는 유연성이 있기 때문에 permutation-invariant property를 따르는 경우에만 transformation 및 score function를 모델링할 수 있다.

따라서, 우리는 신경망을 통해 transformations의 class를 매개 변수화한다.
$$
X :\mbox{a bag of M instances} \\
\varphi_\tau : \mbox{transformer} \ ,\ \ \  \tau : \mbox{parameters} \\
v_{m,k} = \varphi_\tau(x_m), m \in M : \mbox{transforms instances to the embedding space with K dimensions}
$$
그 다음 transformation theta에 의해 x_m의 bag probability가 결정된다.
$$
\theta_\omega : \eta_{\phi_{k\in K}} (v_{m,k}) \to [0,1]
$$
bag level  MIL 풀링 접근법을 사용하는 경우, Theta는 injective function이거나 매개 변수 w를 가진 신경망에 의해 매개 변수화되며, 훈련 가능한 MIL 풀링 방법이 사용되면 phi도 파라미터이다.

### MIL pooling

MIL pooing 
