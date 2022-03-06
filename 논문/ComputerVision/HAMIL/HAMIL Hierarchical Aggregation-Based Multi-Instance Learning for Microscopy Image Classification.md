# HAMIL: Hierarchical Aggregation-Based Multi-Instance Learning for Microscopy Image Classification
## Abstract

Multi-instance learning은 Computer Vision분야, 특히 biomedical image processing 분야에 많이 쓰인다.

기존의 multi-instance learning methods는 feature extraction 또는 학습 단계(learning phase)에서 aggregation operation이 수행되는 feature aggregation methods 및 multi-instance classifiers 설계에 초점을 맞춘다.

본 연구에서는 HAMIL이라는 다중 인스턴스 학습을 위한 hierarchical aggregation network를 제안한다.

hierarchical aggregation protocol은 정의된 순서에 따라 feature 융합을 가능하게 하며, 간단한 컨볼루션 aggregation units는 효율적이고 유연한 아키텍처로 이어진다.

## Introduction

많은 CV Tasks는 multi-instance property을 나타낸다. 예를 들어, 각 입력 데이터 샘플은 bag of instances으로 표현되며 학습은 인스턴스 레벨이 아닌 bag 레벨에서 수행된다.

예를 들어, video-based identity recognition은 각 비디오 입력이 프레임의 시계열이고 각 프레임을 인스턴스로 간주할 수 있기 때문에 multi-instance learning (MIL) 작업으로 처리될 수 있다.

위의 경우 이미지 간에 시간적 또는 공간적 의존성이 있는 반면, 훨씬 더 많은 MIL 시나리오에서는 각 bag 내의 인스턴스들이 정렬되지 않거나 서로 종속성을 보이지 않는다. 이러한 순서가 없는 MIL 작업은 특히 생물의학 이미지 처리에서 일반적이다.

imaging procedure 중에 한 표본에 대한 여러 이미지를 단일 실험 시행으로 캡처하고 반복성을 위해 여러 시행을 수행하는 것이 일반적이다.

유전자의 기능이나 분자의 특성을 유추하기 위해, 단일 이미지가 부분적인 정보만을 포함할 수 있기 때문에 최종 output에 대한 보다 정확한 판단을 제공하기 위해 모든 캡처된 이미지를 포괄적으로 고려해야 한다.

  비디오 또는 3D 데이터와 비교하여 정렬되지 않은 입력을 처리하려면 모델이 순열 불변 속성( permutation-invariant property)을 가져야 한다. 예를 들어 모델은 입력 인스턴스의 표시 순서(presenting order)에 민감하지 않아야 한다. 또한 모델은 가변 크기의 입력과 다양한 이미지 품질로 인해 발생하는 문제를 해결해야 한다.
따라서 MIL 모델을 개발하는 것은 매우 어려운 일이다.

  기존의 MIL 방법은 각각 feature extraction과 classification에 초점을 맞춘 두 가지 범주로 나뉜다.

  메소드의 첫 번째 범주는 classifiers를 공급하기 전에 여러 인스턴스를 고정 길이 입력으로 집계한다.

aggregation operation은 feature extraction 이전 또는 이후에 수행될 수 있다.

FlyIT[9]는 같은 bag에 속한 이미지를 먼저 큰 이미지로 스티치한 다음 큰 이미지에서 feature를 추출하는 전자의 유형이다.

후자 유형은 먼저 원시 인스턴스에 대한 기능을 추출한 다음 집계 작업을 수행한다.

기존 다중 인스턴스 DNN 모델의 성능은 입력 데이터의 복잡성이 높아 한계가 있었다.

이런 문제를 해결하기 위해 HAMIL이라는 새로운 모델을 제시한다.

이 모델은 단순하지만 효과적인 집계 단위를 가진 계층적 집계 프로토콜이 특징이다. 다양한 크기의 input bags을 학습할 수 있을 뿐만 아니라 informative instances를 선호할 수 있다.

우리는 두가지 현미경 이미지 분류 작업에서 HAMIL의 성능을 평가한다.

## RELATED WORK

### 1. Traditional feature aggregation

기존 이미지 처리에서 feature aggregation는 학습 모델에 적용하기 전에 여러 이미지에서 추출한 기능을 포괄적인 기능 표현으로 통합하는 것을 목표로 한다.

세가지 전형적인 feature fusion 방법으로 Bag Of Visual words(BOV), Fisher vector, vector of locally aggregated descriptors (VLAD) 가 있다.

BOV는 image features를 단어로 간주하고, local image features의 vocabulary를 만들고 발생횟수 vector를 생성한다.

Fisher vector 방법은 개별 components의 평균 및 공분산 deviation vectors 뿐만 아니라 Gaussian mixture model (GMM)의 mixing coefficients도 저장한다.

VLAD방법은 각 feature point와 가장 가까운 군집(cluster) 중심 사이의 거리를 계산한다.

이 세 가지 방법 모두 SVM과 같은 기존 머신러닝 모델과 함께 사용할 수 있는 입력 이미지 세트에 대한  fixed-length feature vector를 생성한다.

### 2. **Aggregation of deep convolutional features**

간단한 집계 방법으로서 pooling function는 순열 불변(permutation-invariant)이며 매개 변수가 필요하지 않으므로 MIL 작업에서 널리 사용되어 왔다.

풀링의 주요 결함은 중요한 인스턴스에 집중할 수 없다는 것이지만 attention mechanism은 인스턴스에 점수/가중치를 할당할 수 있기 때문에 좋은 선택이다.

 제안된 HAMIL 모델은 기존과 다른 aggregation mechanism을 갖고 있다.



trainable 및 non-linear convolution aggregation units를 가지고 있으며, permutation-invariance을 위한 hierarchical clustering protocol을 설계한다.

GNN 기반 모델의 계층적 그래프 클러스터링과 달리, 클러스터 수와 집계 시간은 고정 하이퍼 파라미터가 아닌 input bag 크기에 의해 자동으로 결정된다.

## METHODS

### 1. Problem description

이 연구에서, 우리는 더 복잡한 MIL 문제, 즉 다중 인스턴스 다중 라벨 학습에 대해 논의한다.

X : sample set이라 하자. 
$$
X = {X_i}, \ \ \ \ \ \ \ i ∈ \{ 1,2,...,n \} \\
i : \mbox{ \# samples} \\
X_i : \mbox{a sample} \\
X_i = \{x_{i,1}, x_{i,2}, . . . , x_{i,m}\} \\
m  : \mbox{\# instances of i-th sample} \\
x_{i,j} (j \in \{1,2,...,m\}) \mbox{ : an instance of } X_i \\
Y = {Y_i} : \mbox{output space} \\
Y_i = \{ y_1,y_2,...,y_k \} : \mbox{label set of Xi}
$$
목표는 mapping function f : X → Y 를 학습하는 것이다.

특히 이미지 처리 작업에 집중하면서 각 instance는 이미지이고 각 sample은 bag of images으로 표현된다.

### 2. Model Architecture

![Screen Shot 2021-11-30 at 2.29.05 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-11-30 at 2.29.05 PM.png)

크기와 크기가 다양한 input bags를 처리하기 위해 HAMIL은 백의 인스턴스 수에 제한을 두지 않고 계층적 집계 프로토콜을 구현한다.

Feature Extraction, Feature aggregation 및 Classfication 세 가지 주요 구성요소(components)가 있다.

첫번째 component는 Feature Extractor로써 여러개의 convolution layers로 구성된다.

CNN layers에는 입력 인스턴스 집합에 대한 통일된 표현을 생성하기 위한 계층적 집계 절차가 뒤따르며, 이는 분류를 위해 완전히 fully connected layer에 추가로 공급된다.

특히, input bag을 이미지 set로 간주함으로써, HAMIL은 각 bag 내에 이미지/인스턴스의 계층을 구성하여 트리 구조, 즉 리프 노드에서 hierarchical tree의 루트 노드까지 집계 순서를 결정한다.

인스턴스 계층 구조의 구성은 알고리즘 1에 설명되어 있다.

![Screen Shot 2021-11-30 at 2.42.53 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-11-30 at 2.42.53 PM.png)

Line 11에서 두 clusters의 거리는 모든 cross-cluster instance 쌍의 최소 거리, 즉 single-link method로 정의된다. 그리고 인스턴스 간의 거리는 HAMIL의 첫 번째 구성 요소에서 컨볼루션 레이어에 의해 산출된 특징에 기초하여 계산된 유클리드 거리다.

총 K 병합 단계가 있다고 가정하면(K는 하이퍼 파라미터가 아니라 인스턴스 수에 따라 결정되는 이진 트리의 높이), 길이 K의 queue T를 유지한다.

각 배열에 있는 element는 tk로 표시된 triplet이다.
$$
t_k = <I_{C1}, I_{C2}, I_C >, k \in \{ 1, ... , K\}
$$
tk는 S에 있는 군집의 세 가지 인덱스로 구성된다.

처음 두 index는 k번째 단계에서 병합되는 두 클러스터의 인덱스이고, 마지막 한개는 새로 생성된 클러스터의 인덱스이다.

그런 다음, aggregation order는 T의 records에 의해 결정되며, 구체적인 aggregation operations는 다음 Section에서 설명한대로 커널 함수에 의해 정의된다.

### 3. Feature aggregation unit

컨볼루션 연산은 입력의 weighted average로 볼 수 있으므로 컨볼루션을 통해 aggregation units를 설계한다.

세부적으로, 채널을 따라 H × W 행렬의 C 쌍으로 간주될 수 있는 크기 H × W × C의 두 개의 입력 형상 맵이 주어진다. 즉, 각 채널에 대해 행렬 쌍이 있다.

우리는 먼저 각 쌍을 feature aggregation unit에 입력하여 H x W 크기의 aggregated output을 얻는다.

그런 다음, C outputs는 H × W × C 크기의 최종 출력으로 연결(concatenate)된다.

Figure 3은 aggregation units를 보여준다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-11-30 at 3.47.07 PM.png" alt="Screen Shot 2021-11-30 at 3.47.07 PM" style="zoom:50%;" />



Aggregated될 feature maps를 x1, x2라 하자. Aggregation은 다음과 같다.
$$
X = [x_1,x_2],\\
O = W * X + b
$$
X : feature maps로 구성된 tensor, 

`*` : convolution operator

W : convolutional filter, b  : bias

O : aggregated feature map

이것은 One-layer aggregation unit이고, 이것을 L1Agg라고 한다.

더 복잡한 작업의 요구를 충족하기 위해 aggregation units를 여러 계층이 있는 더 깊은 버전으로 확장할 수 있다.

basic aggregation operation을 2번 수행하는 two-layer aggregation units은 다음과 같다.
$$
O = W' * f(g(W * X + b)) + b'
$$
g(·) : normalization function

 f(·) : ReLU

Three-layer aggregation units는 다음과 같다.
$$
O = W'' ∗ f(g(W'∗f(g(W∗X+b))+b'))+b'',
$$
계층 수에 관계없이 aggregation units는 모든 집계 작업에서 공유된다. 따라서, aggregation module은 backbone CNN 모델에 비해 매개 변수의 수가 약간 증가했을 뿐이다.

## EXPERIMENTS

우리는 두 가지 전형적인 종류의 생체 이미지, 즉 미세 세포 이미지와 유전자 발현 이미지를 포함하는 두 가지 대규모 이미지 분류 작업에서 모델 성능을 평가한다. 

### Task I: Prediction of protein subcellular location using im- munofluorescence (IF) images.

 각 단백질은 여러 조직에서 캡처한 현미경 이미지들의 bag에 해당한다.

라벨(즉, 셀룰러 위치)은 이러한 이미지에 내포된 모든 위치 파악 패턴을 기반으로 예측된다. 단백질은 여러 곳에 존재할 수 있다.

### Task II: Gene function annotation using gene expression images.

*in situ* hybridization(ISH) 영상 기술은 조직 내 유전자 발현 공간 분포 패턴을 확인하고 유전자 기능을 드러내는 데 도움을 준다. 

각각의 유전자는 다른 각도나 실험으로 포착된 발현 이미지들의 bag에 해당한다.

명백하게, 이 두 가지 작업은 모두 다중 인스턴스 다중 라벨 분류이다.

우리는 HAMIL을 다음과 같이 single-instance learning models 및 기존 feature aggregation models와 비교한다. HAMIL의 성능을 평가하기 위해 다음을 포함한 7가지 기준 모델과 비교한다.

- A single-instance learning model (SI)1
- 평균 풀링을 사용한 MI, 최대 풀링을 사용한 MVCNN [16] 및 합계 풀링을 사용한 SPoC [23]의 세 가지 풀링 기반 방법
- 두 가지 attention-based method, 즉 NAN[3]과 attention
- 특별히 설계된 Deep MIL 모델인 DeepMIML

모든 baseline 모델에는 원시 이미지에서 features을 추출하기 위한 동일한 백본 네트워크(ResNet18)가 있다. Prediction 성능은 AUC(ROC 곡선 아래의 영역), macro F1 및 micro F1의 세 가지 메트릭으로 평가된다.

