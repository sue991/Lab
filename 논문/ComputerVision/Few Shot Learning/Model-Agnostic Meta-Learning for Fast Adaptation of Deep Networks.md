# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

## Abstract

우리는 gradient descent로 훈련된 모든 모델과 호환되고 분류, 회귀 및 강화 학습을 포함한 다양한 학습 문제에 적용할 수 있다는 점에서 모델에 구애받지 않는 메타 학습 알고리즘을 제안한다.

메타 학습의 목표는 다양한 학습 과제에 대한 모델을 훈련하여 적은 수의 훈련 샘플만 사용하여 새로운 학습 과제를 해결할 수 있도록 하는 것이다.

우리의 접근 방식에서 모델의 parameters는 새로운 작업의 적은 양의 훈련 데이터로 적은 수의 gradient steps가 해당 작업에 대한 우수한 일반화 성능을 산출하도록 명시적으로 훈련된다. 실제로, 우리의 방법은 모델을 미세 조정하기 쉽도록 훈련한다.

우리는 이 접근 방식이 두 개의 퓨샷 이미지 분류 벤치마크에서 최첨단 성능을 제공하고, 퓨샷 회귀에서 좋은 결과를 도출하며, 신경망 정책을 통한 정책 그레이디언트 강화 학습을 위한 미세 조정을 가속화한다는 것을 입증한다.

## Introduction

 Agent가 이전 경험을 적은 양의 새로운 정보와 통합하면서 새로운 데이터에 overfitting되지 않도록 해야 하기 때문에 이러한 종류의 빠르고 유연한 학습은 어렵다. 또한 사전 경험 및 새로운 데이터의 형태는 작업에 따라 달라진다. 따라서 가장 큰 적용 가능성을 위해, 학습을 위한 메커니즘(또는 메타 학습)은 과제를 완료하는 데 필요한 과제와 계산의 형태에 일반적이어야 한다.

본 연구에서는 gradient descent procedure를 통해 훈련된 모든 학습 문제와 모델에 직접 적용할 수 있다는 점에서 일반적이고 모델에 구애받지 않는 메타 학습 알고리즘을 제안한다.

이 방법의 기본이 되는 핵심 아이디어는 모델의 initial parameters를 훈련하여 해당 새 작업에서 적은 양의 데이터로 계산된 하나 이상의 gradient steps를 통해 parameters가 업데이트된 후 새 작업에서 최대 성능을 발휘하도록 하는 것이다.

Update function 또는 learning rule을 학습하는 이전의 메타 학습 방법과 달리, 우리의 알고리즘은 학습된 파라미터의 수를 확장하거나 모델 아키텍처에 제약을 두지 않으며, fully connected, 컨볼루션 또는 recurrent neural networks와 쉽게 결합될 수 있다.

이 작업의 주요 기여는 적은 수의 gradient updates가 새로운 작업에 대한 빠른 학습으로 이어질 수 있도록 모델의 매개 변수를 훈련하는 메타 학습을 위한 간단한 모델 및 task-agnostic  알고리즘이다.

## Model-Agnostic Meta-Learning

우리는 종종 퓨샷 학습으로 공식화되는 문제 설정인 신속한 적응을 달성할 수 있는 모델을 교육하는 것을 목표로 한다. 이 절에서는 문제 설정을 정의하고 알고리즘의 일반적인 형태를 제시하겠다.

### 1. Meta-Learning Problem Set-Up

이 섹션에서는 다양한 학습 영역의 간단한 예를 포함하여 일반적인 방식으로 이 메타 학습 문제 설정을 공식화한다.

`f`라는 모델이 observation `x` 를 output `a` 에 매핑한다고 하자.

분류 학습에서 강화 학습에 이르기까지 다양한 학습 문제에 프레임 워크를 적용하고자 하기 때문에 학습 과제에 대한 일반적인 개념을 소개한다.

각각 task는  loss function L,  initial observations의 분포  q(x1),  transition distribution q(x_t+1|xt,at), episode length H로 구성되어 있다.
$$
T = {L(x_1,a_1,...,x_H,a_H),q(x_1),q(x_{t+1}|x_t,a_t),H}
$$
우리의 meta-learning 시나리오에서, 모델이 적응할 수 있도록 tasks에 대한 분포`p(T)` 를 고려한다.

K-shot learning setting에서, 모델은 qi에서 추출한 K개의 samples로부터의 p(T)에서 추출된 새로운 task Ti을 학습하도록 훈련된다. meta-training동안 task Ti 는 p(T )에서 샘플링되고, model은 K samples와 Ti의 해당 loss L_Ti의 feedback을 사용하여 학습된 다음 Ti의 새 samples에 대해 테스트된다.

그런 다음  파라미터와 관련하여 qi 의 새 데이터에 대한 test error 가 어떻게 변하는지를 고려하여 모델 `f`를 개선한다.

실제로 샘플링된 task의 Ti test error는 메타 학습 프로세스의 training error 역할을 한다.

메타 훈련이 끝나면 새로운 작업이 p(T)에서 샘플링되고, 메타 성능은 K 샘플에서 학습한 후 모델의 성과로 측정된니다.

### 2. A Model-Agnostic Meta-Learning Algorithm

우리는 빠른 적응을 위해 모델을 준비하는 방식으로 메타 학습을 통해 표준 모델의 매개 변수를 학습할 수 있는 방법을 제안한다.

모델은 새로운 과제에 대한 gradient-based learning rule을 사용하여 미세 조정될 것이기 때문에, 우리는 이 gradient-based learning rule이 과적합 없이 p(T )에서 도출된 새로운 과제를 빠르게 진행할 수 있는 방식으로 모델을 학습하는 것을 목표로 할 것이다.

실제로, 우리는 해당 loss의 gradient 방향으로 변경될 때 파라미터의 작은 변경이 p(T )에서 도출된 모든 작업의 손실 함수를 크게 개선할 수 있도록 작업의 변화에 민감한 모델 매개 변수를 찾는 것을 목표로 한다.

일부 매개 변수 벡터 θ에 의해 매개 변수화되고 θ에서 손실 함수가 충분히 부드러워서 그레이디언트 기반 학습 기술을 사용할 수 있다고 가정하는 것 외에는 모델의 형태를 가정하지 않는다.

형식적으로, 우리는 parameters θ를 가진 매개 변수화된 함수 f_θ로 표현되는 모델을 고려한다.

새 Task Ti에 적용할 때, model’s parameters θ는 θ′가 된다.

우리의 방법에서 updated parameter vector θ′i는 task Ti에서 하나 이상의 gradient descent updates를 사용하여 계산된다. 예를 들어 하나의  gradient update를 사용하면 다음과 같다.
$$
θ'_i = θ − α∇_θL_{T_i} (f_θ)
$$
Step size α는 하이퍼 파라미터 또는 메타 학습으로 고정될 수 있다.

모델 파라미터는 p(T)에서 샘플링된 task에 걸쳐 θ에 대한 f_θ'i의 성능을 최적화하여 훈련된다.

구체적으로 meta-objective는 다음과 같다.
$$
\min_\theta \sum_{T_i~p(T)}L_{T_i}(f_{\theta'_i}) = \sum_{T_i~p(T)}L_{T_i}(f_{\theta - \alpha∇_\theta L_{T_i}(f_\theta)})
$$
meta-optimization는 모델 파라미터 θ에 대해 수행되는 반면, objective는 업데이트된 모델 파라미터 θ′에 따라 계산된다.

task 전반에 걸친 meta-optimization은 stochastic gradient descent (SGD)를 통해 수행되며, 모델 파라미터 θ 는 다음과 같이 업데이트된다:
$$
θ←θ - \beta∇_\theta \sum_{T_i ~p(T)} L_{T_i}(f_{\theta'_i})
$$
β : meta step size

일반적인 경우 전체 알고리즘은 다음과 같다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-11-18 at 3.59.11 PM.png" alt="Screen Shot 2021-11-18 at 3.59.11 PM" style="zoom:50%;" />

## Species of MAML

  이 섹션에서는 지도 학습 및 강화 학습을 위한 메타 학습 알고리즘의 특정 인스턴스화에 대해 논의한다. 도메인은 손실 함수의 형태와 작업에 의해 데이터가 생성되고 모델에 사전 전송되는 방법에 따라 다르지만, 두 경우 모두 동일한 기본 적응 메커니즘이 적용될 수 있다.

### 1. Supervised Regression and Classification

메타 학습 정의의 맥락에서 감독 회귀 및 분류 문제를 공식화하기 위해, 모델이 일련의 입력과 출력이 아닌 단일 입력을 받아들이고 단일 출력을 생성하기 때문에 우리는 horizon H = 1을 정의하고 xt에 timestep subscript를 놓을 수 있다.

task  Ti 는 q_i로부터 K i.i.d observations `x` 를 생성하며, task loss는 x에 대한 모델의 output과 target value `y` 사이의 오차로 표시된다.

자주 사용되는 loss 두가지 cross-entropy, mean squared error(MSE) 가 있다.

**MSE**는 다음과 같다.
$$
L_{T_i} (f_\phi) = \sum_{\mathbf{x}^{(j)},\mathbf{y}^{(j)} \sim T_i} |
|f_\phi (\mathbf x^{(j)}) - \mathbf y^{(j)}||^2_2
$$
K -shot regression tasks에서, K개의 input/output 쌍은 각 task 별 learning을 위해 제공된다.

**cross- entropy loss** 는 다음과 같다.
$$
L_{T_i}(f_\phi) = \sum_{\mathbf x^{(j)}, \mathbf y^{(j)} \sim T_i} \mathbf y^{(j)} \log f_\phi (\mathbf x^{(j)}) + (1 - \mathbf y^{(j)}) \log {(1 - f_\phi ( \mathbf x^{ (j)} )} )
$$

### 2. Reinforcement Learning

새로운 task에는 새로운 목표를 달성하거나 새로운 환경에서 이전에 훈련된 목표를 달성하는 작업이 포함될 수 있다.

예를 들어, 에이전트는 새로운 미로에 직면했을 때 몇 개의 샘플만으로 안정적으로 출구에 도달할 수 있는 방법을 결정할 수 있도록 미로를 빠르게 탐색하는 방법을 배울 수 있다. 

이 섹션에서는 RL을 위한 메타러닝에 MAML을 적용하는 방법에 대해 논의하겠다.

메타 학습에 대한 또 다른 접근법은 반복 학습자가 새로운 작업에 적응하도록 훈련하는 많은 작업에서 기억 증강 모델을 훈련하는 것이다.

똫나 우리의 접근 방식은 단순히 좋은 weight initialization을 제공하고 학습자와 메타 업데이트 모두에 대해 동일한 gradient descent 업데이트를 사용한다. 따라서 추가적인 경사도 단계를 위해 학습자를 쉽게 미세 조정할 수 있다.

컴퓨터 비전에서, 대규모 이미지 분류에 사전 훈련된 모델은 다양한 문제에 대한 효과적인 특징을 학습하는 것으로 나타났다.

대조적으로, 우리의 방법은 빠른 적응성을 위해 모델을 명시적으로 최적화하여 몇 가지 예만으로 새로운 작업에 적응할 수 있도록 한다.

또한 우리의 방법은 모델 매개 변수에 대한 새로운 작업 손실의 민감도를 명시적으로 최대화하는 것으로 볼 수 있다. 많은 선행 연구들이 종종 초기화 맥락에서 심층 네트워크에서 민감성을 시험했다.

대조적으로, 우리의 방법은 주어진 작업 분포에서 민감도에 대한 매개 변수를 명시적으로 훈련하여 한 단계 또는 몇 단계의 경사도에서 K-shot 학습 및 신속한 강화 학습과 같은 문제에 매우 효율적으로 적응할 수 있도록 한다.

## Experimental Evaluation

실험 평가의 목표는 다음 질문들에 대답하는 것이다.

(1) MAML이 새로운 task의 빠른 학습을 가능하게 할 수 있나?

(2) MAML이 감독된 회귀, 분류 및 강화 학습을 포함한 여러 다른 영역의 메타 학습에 사용될 수 있나?

(3) MAML로 학습한 모델은 추가 그라데이션 업데이트 및/또는 예를 통해 지속적으로 개선될 수 있나?

가능한 경우, 우리는 결과를 모델의 성능에 대한 상한으로 task의 identity(문제 의존적 표현)를 추가 입력으로 수신하는 오라클과 비교합니다.

### 1. Regression

각 작업에는 사인파의 입력에서 출력으로 회귀하는 과정이 포함되며, 사인파의 진폭과 위상이 작업 간에 달라진다.

따라서 p(T )는 연속이며, 여기서 진폭(amplitude)은 [0.1, 5.0] 내에서 변화하고 phase은 [0, [] 내에서 변화하며 입력과 출력은 모두 1의 치수성을 갖습니다.

 Training and testing동안, datapoints **x**는 [-5.0,5.0]에서 균일하게 샘플링된다. loss는  mean-squared error를 사용한다. 

### 2. Classification

Omniglot dataset : 50가지의 언어에 대해 총 1623개의 문자가 20명의 다른 사람에게서 쓰인 데이터셋.  즉, 총 20*1623개의 데이터 존재

MiniImagenet dataset : 64 training class, 12 valid class, 24 test class (ImageNet dataset의 축소 버전이라고 생각)

우리는 Vinyals 등(2016)이 제안한 실험 프로토콜을 따르는데, 이 프로토콜은 1개 또는 5개의 샷으로 N-way Classification을 빠르게 학습하는 것을 포함한다.

N-way classification 문제는 다음과 같이 설정된다: 보이지 않는 N 클래스를 선택하고, 각 N 클래스의 K개의 다른 인스턴스로 모델을 트레이닝하며, N 클래스 내에서 새로운 instances를 분류하는 모델의 능력을 평가한다.

Omniglot에서, 알파벳에 상관없이 랜덤하게 1200개 characters를 training에 사용하고, 나머지를 testing에 사용한다. Omniglot 데이터 세트는  90도 회전으로 augmentation된다.



MAML에서 상당한 계산 비용은 메타 목적의 그레이디언트 연산자를 통해 메타 그레이디언트를 역전파할 때 이차 도함수를 사용함으로써 발생한다



### 3. Reinforcement Learning





