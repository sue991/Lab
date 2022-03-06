# Prototypical Networks for Few-shot Learning(2017)

## Abstract

> Training set에 있지 않은 새로운 class에 대해 학습할 때, 그 새로운 class에 대한 examples 수가 적은 경우인 few-shot classification 문제를 위한 *prototypical network* 를 제안한다.

Prototypical network는 각 class의 prototype representation의 거리를 계산하여 classification을 수행할 수 있는 metric space를 학습한다.

최근 Few-shot learning 접근법과 비교했을 때, 제한된 데이터 체계에서 유익하고 보다 단순한 유도 편향을 반영하여 우수한 결과를 얻었다.

우리는 복잡한 아키텍처 선택과 메타 학습을 포함하는 최근의 접근 방식에 비해 일부 간단한 design decisions이 상당히 개선될 수 있음을 보여주는 분석을 제공한다. 또한 프로토타입 네트워크를 zero-shot learning으로 확장하고 CU-Birds dataset에서 최첨단 결과를 달성한다.

## Introduction

Few-shot classification은 train dataset에 없는 새로운 class에 대한 적은 데이터만을 가지고 있을 경우, Classifier가 그 class를 수용하기 위해 조정되야 하는 task이다.

기본 접근은 모델을 새로운 데이터로 re-training하는 것이지만, overfitting 가능성이 높다. 문제가 매우 어려운 반면, 인간은 높은 정확도로 각 새로운 class의 단 하나의 example만 주어지는 one-shot classification을 할 수 있다.

두개의 최신 연구는 few-shot learning에서 상당한 진보를 보여주었다. 

Vinyals et al.은 *matching networks*를 제안하였는데, 이것은 unlabeled point(query set)에 대한 class를 예측하기 위해 labeled set(support set)이 학습된 embedding에 attention mechanism을 사용한다.

Matching network는 한 embedding space내에서 적용되는 weighted nearest-neighbor classifier로 해석될 수 있다. 이 모델은 training 동안 *episodes*라 불리는 ampled mini-batches를 사용한다. 이때, 각 episode는 데이터 포인트뿐만 아니라 classes를 subsampling하여 few-shot task를 모방하도록 설계되었다. 

Episodes의 사용은 training 문제가 테스트 환경에 더욱 충실해져 일반화가 향상된다.

Ravi and Larochelle은 episodic training idea를 더 나아가 few-shot learning에 대한 메타 학습 방식을 제안한다.

이러한 접근 방식에는 LSTM이 테스트 세트에 잘 일반화될 수 있도록 episode가 주어진 경우 classifier에 대한 업데이트를 생성하도록 교육하는 것이 포함된다. 여기서 LSTM 메타 학습자는 여러 에피소드에 걸쳐 단일 모델을 교육하는 대신 각 에피소드에 맞는 맞춤형 모델을 교육하는 방법을 배운다.

저자는 이 둘의 문제점인 overfitting을 지적하고 이를 줄이기 방향으로 protypical networks를 고안하였다.

데이터가 매우 제안되어있기 때문에, 매우 단순한 inductive bias를 가져야 한다는 가정 하에 작업했다.

Protonet은 각 class에 대해 single prototype representation이 있는 embedding을 base로 접근하였다.

이를 위해 neural network를 사용하여 임베딩 공간에 대한 input의 non-linear mapping을 학습하고, 임베딩 공간에서 설정된 support set의 평균으로 class의 prototype을 만든다.

그 다음 Classification을 수행할 때 임베딩 된 query point에서 가장 가까운 class prototype을 찾는다.

> Zero-shot learning에서도 같은 방식의 접근법을 가진다. 여기서는 각 클래스가 레이블링된 소수의 example이 아닌 높은 수준의 클래스 설명을 제공하는 메타 데이터와 함께 제공된다.

따라서 protonet은 각 class의 prototype 역할을 하기 위해 meta-data를 공유 공간(shared space)에 임베딩하는 것을 학습한다.

Classification은 few-shot scenario에서와 같이 내장된 쿼리 포인트에 가장 가까운 클래스 프로토타입을 찾아 수행한다.

>Classification을 수행할 때 임베딩 된 query point에서 가장 가까운 class prototype을 찾는다. 각 class의 평균으로 prototype을 만들고 Euclidean distance를 이용해서 query point와의 거리를 계산한다. 이 거리 중 가장 가까운 prototype을 결정하고 query point의 class를 해당 prototype의 class로 예측한다.

![Screen Shot 2021-09-13 at 3.10.53 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-09-13 at 3.10.53 PM.png)

이 논문에서, 우리는 퓨샷 및 제로샷 설정 모두에 대한 프로토타입 네트워크를 공식화한다.

우리는 one-shot setting에서 matching networks에 대한 connection을 그리고, 모델에 사용된 underlying distance function를 분석한다.

특히, 거리가 squared Euclidean distance와 같은 Bregman divergence로 계산될 때, class 평균을 prototype으로 사용하는 것을 정당화하기 위해 prototypical network를 clustering과 관련시킨다.

우리는 Euclidean distance가 더 일반적으로 사용되는 코사인 유사성을 크게 능가하기 때문에 distance 선택이 필수적이라는 것을 경험적으로 발견하였다. 몇 가지 벤치마크 작업에서 우리는 최첨단 성능을 달성한다. 프로토타입 네트워크는 최근의 메타 학습 알고리즘보다 간단하고 효율적이므로 퓨샷 및 제로샷 학습에 대한 매력적인 접근 방식이 된다.

## Prototypical Networks

### 1. Notation

Few-shot classification에서는 N개의 labeled된 support set example S={(x1,y1),...,(xN,yN)}가 존재하고 xi∈RD는 D-dimensional feature vector of example이고, yi∈{1,...,K}는 corresponding label이다. Sk는 class k에 대한 dataset을 말한다. 
$$
S = \{(x_1, y_1), . . . , (x_N , y_N)\} \\
x_i \in \mathbb{R}^D : \mbox{D-dimensional feature vector of an example} \\
y_i ∈ \{1, . . . , K\} : \mbox{corresponding label} \\
S_k  : \mbox{set of examples labeled with class k}
$$


### 2. Model

Prototypical networks는 M-dimensional representation인 ck ∈ RM or prototype을 계산하고, 

각각의 class는 embedding function fϕ:RD→RM을 거친다. ϕ는 learnable parameter(weight)이다. 각각의 prototype은 각 class에 속한 embedded support points의 mean vector이다.
$$
\mathbf{c}_k \in \mathbb{R}^M \mbox{or prototype}\\
f_\phi : \mathbb{R}^D → \mathbb{R}^M \\
\phi : \mbox{learnable parameters} \\

\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(x_i,y_i) \in S_k} f_\phi(x_i)
$$
distance function `d`가 주어졌을 때, 
$$
d : \mathbb{R}^M \times \mathbb{R}^M \to [0,+\infty),
$$
prototypical networks는 embedding space에서의 prototype에 대한 distribution을 생성해 내는데, 이 distribution은 distance로 softmax한 query point `x`의 class를 결정할 때 필요하다.
$$
p_\phi(y = k|\mathbf{x}) = \frac{\exp(-d(f_\phi(\mathbf{x}),\mathbf{c}_k))}{\sum_{k'} \exp(-d(f_\phi(\mathbf{x}),\mathbf{c}_{k'}))}
$$

>protonet은 embedding space에서 각 class를 대표하는 prototype의 분포를 생성하고(위 그림에서 C1,C2 등등) query dataset 중 하나인 query point x를 distance 기반의 softmax를 취한 값을 비교할 때 사용한다.

Negative log-probability 
$$
J(ϕ)= −\log p_ϕ(y=k|\mathbf{x})
$$
를 최소화 하기 위해 SGD를 이용하고, Training episode는 training set에서 랜덤하게 class를 선택하여 만든다. 그리고 남은 것 중 일부를 선택하여 query point를 만든다. 

Algorithm 1에 episode를 training하기 위해  loss J(φ)를 계산하는 수도코드가 나와있다.

![Screen Shot 2021-09-13 at 4.01.14 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-09-13 at 4.01.14 PM.png)

### 3. Prototypical Networks as Mixture Density Estimation

 *Regular Bregman divergences*라고 알려진 distance function에 대해서, prototypical networks algorithm은 support set에 대해 performing mixture density estiation을 적용한다. Regular Bregman divergence `dφ`는 다음과 같이 정의된다.
$$
d_φ(\mathbf{z},\mathbf{z}') = φ(\mathbf{z}) - φ(\mathbf{z}') - (\mathbf{z}-\mathbf{z}')^T \nabla φ(\mathbf{z}') \\
φ : \mbox{ditterentiable(미분가능한), strictly convex function of the Legendre type}
$$
Bregman divergences는 squared Euclidean distance ||z−z′||2 와 Mahalanobis distance 또한 포함한다.

Prototype computation은 support set의 하드 클러스터링 측면에서 볼 수 있으며, 클래스당 하나의 클러스터와 각 지원 지점이 해당 클래스 클러스터에 할당된다. Bregman divergences의 경우 할당된 지점까지의 최소 거리를 달성하는 대표적인 군집이 cluster mean이라는 것이 나타났다.

따라서 방정식 (1)의 프로토타입 계산에서는 브레그만 분기가 사용될 때 지원 세트 레이블이 주어진 최적의 군집 대표자를 산출한다.



