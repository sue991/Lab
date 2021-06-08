# Neural Collaborative Filtering

## ABSTRACT

DNN이 speech recognition, CV, NLP등 에서 엄청난 성과를 거두었다. 그러나 recommender system에서의 DNN은 상대적으로 덜 정밀한 조사?를 받았다. 이 연구에서 implicit feedback을 기반으로 recommendation의 핵심 문제인 collaborative filtering을 해결하기 위해 NN을 이용한 기술을 개발하기 위해 노력한다.

비록 몇몇 최근 연구가 딥러닝을 recommendation에 사용해왔지만, 그들은 주로 아이템의 textial descriptions와 음악의 acoustic features과 같은 보조 정보를 모델링하는데 사용했다.

collaborate filtering에서 가장 중요한 요소인 user와 item 간의 interaction을 모델링 하는 것에 관해, 그들은 여전히 MF에 의존했고 user 와 item의 latent features에 inner product를 적용했다. 

inner product를 데이터로부터 임의의 기능을 학습할 수 있는 neural architecture 로 대체하면서, 우리는 NCF라는 general한 framwork를 제시한다. NCF는 일반적이며 그 프레임워크 하에서 MF를 표현하고 일반화할 수 있다. 비선형성으로 NCF 모델링을 슈퍼차지하기 위해 MLP를 활용하여 사용자-항목 상호 작용 기능을 학습할 것을 제안한다. 두 개의 실제 데이터 세트에 대한 광범위한 실험은 제안된 NCF 프레임워크가 최첨단 방법에 비해 크게 개선되었음을 보여준다. 경험적 증거는 신경망의 더 깊은 계층을 사용하는 것이 더 나은 권장 성능을 제공한다는 것을 보여준다.

## 1. INTRODUCTION

정보 폭발의 시대에서 recommender systems는 전자상거래, 온라인 뉴스 및 소셜 미디어 사이트를 포함한 많은 온라인 서비스에 의해 널리 채택되어 온 정보 과부하를 완화하는 데 중추적인 역할을 한다. 개인적인 recommender system의 핵심은 collaborative filtering으로 알려져있는 그들의 과거 interaction(e.g. ratings and clicks)에 기반한 items에서 user의 선호도를 모델링하는 것에 있다. 다양한 collaborative filtering 중에서 MF 가 가장 인기 있는 것으로, 사용자 또는 항목을 나타내는 latent features의 벡터를 사용하여 사용자와 항목을 공유 latent space에 투영한다. 그 후에 한 item에 대한 user의 interaction은  latent vector의 inner product로 모델링된다.

Netflix Prize에 의해 대중화된 MF는 latent factor model-based recommendation에 대한 사실상의 접근법이 되었다. MF를 neighbor-based 모델과 통합하고, item content의 topic 모델[38]과 결합하고, features의 일반적인 모델링을 위해 factorization machine으로 확장하는 것과 같은 많은 연구가 이루어졌다. Collaborate filtering에 대한 MF의 효과에도 불구하고, interaction function인 inner product의 단순한 선택에 의해 성능이 방해될 수 있다는 것은 잘 알려져 있다. 예를들어 explicit feedback을 이용한 rating 예측 task에서, user와 item bias terms를 interaction function에 통합함으로써 MF 모델의 성능을 향상시킬 수 있다는 것은 잘 알려져 있다. inner product operator에게는 사소한 수정일 뿐이지만, 사용자와 item 사이의 latent feature interactions를 모델링하기 위한 더 나은 전용 interaction function을 설계하는 긍정적인 효과를 나타낸다. 단순히 latent features의 곱셈을 linear하게 결합하는 inner product는 user interaction data의 복잡한 구조를 포착하기에 충분하지 않을 수 있다.

본 논문은 많은 이전 연구에서 수행된 수작업이 아닌 데이터로부터 interaction function을 학습하기 위한 DNN의 사용을 탐구한다. NN은 continuous function을 거의 정확하게 근사할 수 있는 것으로 입증되었으며, 보다 최근에는 DNN이 컴퓨터 비전, 음성 인식에서 텍스트 처리에 이르는 여러 영역에서 효과적인 것으로 밝혀졌다. 그러나 MF 방법에 대한 방대한 양의 문헌과 대조적으로, 추천을 위해 DNN을 사용하는 작업은 상대적으로 적다. 최근 몇몇 연구는 DNN을 추천 작업에 적용하고 유망한 결과를 보여주었지만, 그들은 대부분 DNN을 사용하여 항목의 텍스트 설명, 음악의 오디오 특징 및 이미지의 시각적 내용과 같은 보조 정보를 모델링했다. 핵심 collaborative filtering 효과 모델링과 관련하여, 그들은 여전히 MF에 의존하여 inner product를  사용하여 사용자와 item의 latent features을 결합했다.

이 연구는 collaboratice filtering을 위한 신경망 모델링 접근법을 공식화하여 앞서 언급한 연구 문제를 해결한다. 우리는 비디오를 보고, 제품을 구매하고, 아이템을 클릭하는 등의 행동을 통해 유저의 선호를 간접적으로 확인하는 implicit feedback에 초점을 맞춘다. Explicit feedback(rating ans reviews)와 비교했을 때, implicit feedback은 자동적으로 추적될 수 있고 따라서 더 content 제공자로부터 모으기가 쉽다. 그러나, 이것은 유저 만족감이 제공되지 않고 부정적이 feedback이 자연히 부족하기에 활용하는데 있어서 더 도전적이다.

본 논문에서는 DNN을 활용하여 노이즈가 많은 implicit feedback 신호를 모델링하는 방법에 대한 중심 주제를 탐구한다.

이 일의 주요 기여는 다음과 같다.

1. 우리는 user와 item의 latent features를 모델링하고 NN 기반의 collaborative filtering을 위한 일반적인 framework NCF를 고안하기 위한 NN 아키텍처를 제시한다.
2. MF가 NCF의 특별한 케이스로 해석될 수 있고 MLP를 활용하여 NCF 모델링에 높은 수준의 비선형성을 제공할 수 있음을 보여준다.
3. 우리는 두 개의 실제 데이터 세트에 대한 광범위한 실험을 수행하여 NCF 접근 방식의 효과와 collaborative filtering에 대한 딥 러닝의 가능성을 입증한다.

## 2. PRELIMINARIES

먼저 문제를 공식화하고 implicit feedback을 통해 collaborative filtering을 위한 기존 솔루션에 대해 논의한다. 그런 다음 널리 사용되는 MF 모델을 간략히 요약하여 inner product 사용으로 인한 한계를 강조한다.

### 2.1 Learning from Implicit Data

M, N : number of users and items

user-item matrix : 
$$
Y \in R^{M * N}
$$
user's implicit feedback,
$$
y_{ui} =
\begin{cases}
1, & \mbox{if interaction (user } u,\mbox{ item }i)\mbox{ is ibserved;} \\
0, & \mbox{otherwise. }
\end{cases}
$$
여기서 yui의 값이 1이면 user u와 item i 사이에 상호 작용이 있음을 나타냅니다. 그러나 실제로 i를 좋아하는 것은 아닙니다. 마찬가지로 값이 0이라고 해서 반드시 i를 좋아하지 않는 것은 아니며, user가 해당 item을 인식하지 못하는 것일 수 있습니다. 이는 사용자의 선호도에 대한 시끄러운 신호만 제공하기 때문에 implicit 데이터에서 학습하는 데 문제를 제기한다. 관찰된 항목은 최소한 항목에 대한 사용자의 관심을 반영하지만, 관찰되지 않은 항목은 단지 누락된 데이터일 수 있으며 부정적인 피드백의 자연적 부족이 있다.

implicit feedback의 recommendation problem은 item 순위를 매기는 데 사용되는 Y에서 관찰되지 않은 item의 점수를 추정하는 문제로 공식화된다. 모델 기반 접근 방식은 데이터가 기본 모델에 의해 생성(또는 설명)될 수 있다고 가정한다. 공식적으로 그들은 이렇게 추상화될 수 있다.
$$
\hat{y}_{ui} = f (u, i|Θ),
$$


여기서 ^y_ui는 y_ui의 예측점수를 나타내고, Θ는 model parameter를, f는 model parameters를 predict score에 매핑하는 함수(interaction function)를 나타낸다.

Θ를 추정하기 위해, 기존 접근 방식은 일반적으로 objective function을 최적화하는 기계 학습 패러다임을 따른다. 문헌에서 가장 많이 이용되는 두가지 objective functions는 pointwise loss, pairwise loss 이다. implicit feedback에 대한 풍부한 연구의 자연스러운 확장으로서, pointwise learning에 대한 방법은 일반적으로 ^y_ui와 목표값 yui 사이의 squared loss을 최소화함으로써 회귀 프레임워크를 따른다. 네거티브 데이터의 부재를 처리하기 위해, 그들은 관찰되지 않은 모든 항목을 네거티브 피드백으로 처리하거나 관찰되지 않은 항목에서 네커티브 인스턴스를 샘플링했다. pairwise learning의 경우,  관찰된 항목은 관찰되지 않은 항목보다 높은 순위를 매겨야 한다는 생각이다. 이와 같이, pairwise learning은 ^y_ui와 yui 사이의 손실을 최소화하는 대신, 관찰된 항목 ^y_ui와 관찰되지 않은 항목 ^y_ui 사이의 여백을 최대화한다.

한 단계 앞으로 나아가는 우리의 NCF 프레임워크는 NN을 사용하여 interaction function `f`를 매개 변수화하여 ^y_ui를 추정한다. 따라서 pointwise 와 pairwise learning을 모두 자연스럽게 지원한다.

### 2.2 Matrix Factorization

MF는 각 사용자와 항목을 latent features의 실제 값 벡터와 연관시킨다.

p_u , q_i : latent vector for user u and item i

MF는 interaction y_ui 를 pu, qi의 inner product라고 추정한다.
$$
\hat{y}_{ui} = f(u, i|p_u, q_i) = p^T_u q_i = \sum\limits_{k=1}^K p_{uk}q_{ik}, \mbox{ K: latent space의 dimension}
$$


우리가 볼 수 있듯이, MF는 latent space의 각 차원이 서로 독립적이고 동일한 weight로 선형적으로 결합한다고 가정하여 사용자와 item latent factors의 양방향 상호 작용을 모델링한다.

그림 1은 inner product functino이 어떻게 MF의 표현성을 제한할 수 있는지를 보여줍니다. 예시를 잘 이해하기 위해 사전에 두 가지 설정을 명확히 설명해야 합니다. 첫째, MF는 사용자와 항목을 동일한 latent space에 매핑하기 때문에, 두 사용자 간의 유사성도 inner product 또는 동등하게 잠재 벡터 사이의 각도의 코사인으로 측정할 수 있다. 두번째, 일반성의 손실 없이, 우리는 MF가 복구해야 하는 두 사용자의 실제 유사성으로 Jaccard coefficient를 사용한다.

* Jaccard coefficient : 

$$
s_{ij} =\frac{|Ri|∩|Rj|}{|Ri |∪|Rj |} ;  R_u : \mbox{user u가 본 items의 집합}
$$

$$
s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)
$$



이와 같이, latent space에서 p1, p2, p3의 기하학적 관계는 그림 1b와 같이 플로팅될 수 있다. 이제 그림 1a에서 파선으로 입력되는 새로운 사용자 u4를 고려해보자.
$$
s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)
$$
이것은 u4 가  u1과 가장 비슷하고, 다음으로 u3 마지막으로 u2 순서이다.

그러나 MF 모델이 p4를 p1에 가장 가깝게 배치하면(두 가지 옵션은 그림 1b에 점선으로 표시), p4가 p3보다 p2에 더 가깝고, 불행히도 큰 ranking loss가 발생한다.

위의 예는 저차원 latent space에서 복잡한 사용자-item interactions를 추정하기 위해 단순하고 고정된 inner product를 사용함으로써 야기되는 MF의 가능한 한계를 보여준다. 우리는 이를 해결하는 한 방법으로 매우 큰 latent factors K를 사용하는 것에 주목한다. 그러나 특히 sparse setting에서 모델의 일반화(예: 데이터 과적합)에 악영향을 미칠 수 있다. 본 연구에서는 데이터로부터 DNN을 사용하여 interaction function를 학습하여 한계를 해결한다.

## 3. NEURAL COLLABORATIVE FILTERING

먼저 일반적인 NCF 프레임워크를 제시하여 implicit data의 binary property을 강조하는 확률적 모델로 NCF를 학습하는 방법을 자세히 설명한다. 그런 다음 우리는 MF가 NCF에서 표현되고 일반화될 수 있음을 보여준다. collaborative filtering을 위한 DNN을 탐색하기 위해 다중 계층 퍼셉트론(MLP)을 사용하여 user-item interaction function을 학습하는 NCF 인스턴스화를 제안한다. 

### 3.1 General Framework

collaboratice filtering의 full neural treatment를 허용하기 위해, 우리는 그림 2에 나온 것처럼 사용자-항목 상호 작용 y_ui를 모델링하기 위해 multi layer representation을 채택하며, 여기서 한 계층의 출력이 다음 계층의 입력으로 작용한다. 하단 input layer는 각각 사용자 u와 항목 i를 설명하는 두 가지 특성 벡터 v^U_u와 v^I_i로 구성되며, context-ware, content-based 및 neighbor-based와 같은 광범위한 사용자 및 항목의 모델링을 지원하도록 사용자 정의할 수 있다. 본 연구는 순수한 collaborative filtering 설정에 중점을 두기 때문에, 우리는 사용자와 항목의 ID만 입력 기능으로 사용하며, 이를 원-핫 인코딩으로 binarized sparse vector로 변환한다. 입력에 대한 일반적인 기능 표현을 사용하면 content function을 사용하여 사용자와 item을 표시함으로써 cold-start problem를 해결하도록 우리의 방법을 쉽게 조정할 수 있다.

* cold-start problem : 고객의 수가 적어서 고객 맞춤 상품이나 서비스 개발에 활용할 고객 데이터가 부족한 문제

input layer 위에는 임베딩 계층이 있으며, sparse representation을 dense vector에 투영하는 fully connected layer이다. 획득한 사용자(항목) 임베딩은 latent factor model의 context에서 사용자(항목)에 대한 latent vector로 볼 수 있다. 그런 다음 사용자 임베딩과 항목 임베딩은 우리가 neural collaborative filtering layers라고 부르는 multi-layer neural architecture로 공급되어 latent vector를 예측 점수에 매핑한다. neural CF layers의 각 레이어를 커스터마이징하여 user–item interactions의 특정 latent structures를 발견할 수 있다. 마지막 hidden layer X의 dimension에 따라 모델의 기능이 결정됩니다. 최종 output layer는 예측 점수 ^yui이며, ^yui와 목표 값 yui 사이의 pointwise loss를 최소화하여 훈련을 수행한다. 우리는 모델을 훈련시키는 또 다른 방법은 Bayesian Personalized Ranking과 margin-based loss을 사용하는 것과 같은 pairwise 학습을 수행하는 것이라는 점에 주목한다. 논문의 초점이 신경망 모델링 부분에 있기 때문에, 우리는 NCF의 pairwise 학습에 대한 확장을 향후 연구로 남겨둔다. 

우리는 이제 다음과 같이 NCF의 예측 모델을 공식화한다.
$$
\hat{y}_{ui} =f(P^Tv^U_u,Q^Tv^I_i|P,Q,Θ_f) ;
$$

$$
P ∈ R^{M×K} \mbox{ and } Q ∈ R^{N×K} \mbox{ latent factor matrix for users and items}
$$

$$
Θ_f : \mbox{model parameters of the interaction function f}
$$

함수 f는 multi-layer neural network로 정의되므로 다음과 같이 공식화할 수 있다.
$$
f(P^T v^U_u ,Q^T v^I_i ) = \phi_{out}(\phi_X(...\phi_2(\phi_1(P^T v^U_u ,Q^T v^I_i ))...)),
$$

$$
\phi_{out} , \phi_x : \mbox{ output layer와 x-th neural collaborative filtering(CF) 계층에 대한 매핑 함수}
$$

총 X개의 신경 CF 계층이 있다.

### *3.1.1 Learning NCF*

모델 매개 parameters를 학습하기 위해 기존 pointwise은 주로 손실 제곱(squared loss)을 사용한 회귀 분석을 수행합니다.
$$
L_{sqr} = \sum\limits_{(u,i)∈Y ∪Y −} w_{ui}(y_{ui} −\hat{y}_{ui})^2,
$$
Y : Y에서 관찰된 interaction의 집합 

Y− : negative instance의 집합(관찰되지 않은 모든interactions) 

w_ui : training instance (u,i) 의 weight를 나타내는 hyperparameter

squared loss는 관측치가 Gaussian distribution에서 생성된다고 가정하여 설명할 수 있지만, implicit 데이터와 잘 집계되지 않을 수 있다는 점을 지적한다. 이는 implicit 데이터의 경우 target value yui가 u와 i가 상호 작용했는지 여부를 나타내는 이진화 1 또는 0이기 때문이다. 이어지는 내용에서, 우리는 implicit 데이터의 이진 속성에 특별한 주의를 기울이는 pointwise NCF를 학습하기 위한 확률론적 접근법을 제시한다.

implicit 피드백의 단일 클래스 특성을 고려할 때, 우리는 yui의 value를 레이블로 볼 수 있다 - 1은 항목 i가 u와 관련이 있다는 것을 의미하고, 0은 그렇지 않다는 것을 의미한다.

prediction score yˆui 는 i와 u가 얼마나 관련 있을 것 같은지를 표현한다. NCF에 그러한 확률론적 설명을 부여하기 위해, 우리는 [0,1]의 범위에서 출력 ^y_ui를 제한해야 하며, 이는 output layer \phi_out 의 활성화 함수로 확률적 함수(예: 로지스틱 또는 프로빗 함수)를 사용하여 쉽게 달성할 수 있다.
$$
p(Y,Y−|P,Q,Θ_f)=  \prod _{(u,i)∈Y} \hat{y}^{ui} \prod_{(u,j )∈Y −}(1-\hat{y}^{uj})
$$
negative logarithm을 취하면, 
$$
L=− \sum_{(u,i)∈Y} \log \hat{y}_{ui} − \sum_{(u,j)∈Y-} \log(1-\hat{y}_{uj})   \\
= -\sum_{(u,i)∈Y ∪Y −} y_{ui}\log\hat{y}_{ui} + (1-y_{ui})\log(1−\hat{y}_{ui}) .
$$
이는 NCF 방법에 대해 최소화하는 목적 함수이며, 그 최적화는 확률적 경사 하강(SGD)을 수행하여 수행할 수 있다. 신중한 독자들은 그것이 log loss라고도 알려진 binary cross-entropy loss와 같다는 것을 깨달았을지도 모른다. NCF에 대한 확률적 처리를 채택함으로써, 우리는 이진 분류 문제로 implicit 피드백을 가진 권고 사항을 해결한다. classification-aware log loss는 권장 문헌에서 거의 조사되지 않았기 때문에, 우리는 이 연구에서 그것을 탐색하고 4.3절에 그것의 효과를 경험적으로 보여준다. negative instance Y- 의 경우, 우리는 각 반복에서 관찰되지 않은 interactions에서 균일하게 그것들을 샘플링하고 관찰된 interactions 횟수와 관련하여 샘플링 비율을 제어한다. 균일하지 않은 샘플링 전략(예: 항목 인기 편향)은 성능을 더욱 향상시킬 수 있지만, 우리는 탐사를 미래 작업으로 남겨둔다.

### 3.2 Generalized Matrix Factorization (GMF)

우리는 이제 MF가 어떻게 NCF framework의 스페셜 케이스로 해석될 수 있는지 본다. MF는 추천용으로 가장 인기 있는 모델이며 문헌에서 광범위하게 조사되어 왔기 때문에 복구할 수 있으면 NCF가 많은 factorization 모델을 모방할 수 있다.

input layer의 사용자(항목) ID의 one-hot 인코딩으로 인해 얻은 임베딩 벡터는 사용자(항목)의 latent 벡터로 볼 수 있다.
$$
p_u = P^Tv^U_u : \mbox{user latent vector} \\
q_i = Q^Tv^I_i : \mbox{item latent vector}
$$
우리는 첫번째 neural CF layer의 매핑 함수를 
$$
\phi_1(p_u, q_i) = p_u ⊙ q_i, \mbox{ ⊙ : element-wise product of vectors}
$$
라고 정의한다. 그런 다음 벡터를 output layer에 투영한다.
$$
\hat{y}_{ui} = a_{out}(h^T(p_u ⊙ q_i))
$$
a_out : activation function ;  h : edge weights of the output layer,

직관적으로 만약 우리가 a_out에 대해 아이덴티티 함수를 사용하고 h를 1의 통일된 벡터로 시행한다면, 우리는 MF 모델을 정확하게 복구할 수 있다.

NCF 프레임워크에서는 MF를 쉽게 일반화하고 확장할 수 있다. 예를 들어, 균일한 제약 조건 없이 데이터로부터 h를 학습할 수 있게 하면 latent dimensions의 다양한 중요성을 허용하는 MF의 변형을 초래할 것이다. 그리고 우리가 a_out에 비선형 함수를 사용할 경우, 그것은 선형 MF 모델보다 더 표현력이 높을 수 있는 비선형 설정으로 MF를 일반화한다. 본 연구에서는 시그모이드 함수를 a_out으로 사용하고 log loss가 있는 데이터에서 h를 학습하는 MF의 일반 버전을 NCF에서 구현한다. 우리는 그것을 GMF라고 부른다.

### 3.3 Multi-Layer Perceptron (MLP)

NCF는 사용자와 항목을 모델링하기 위해 두 경로를 채택하므로, 두 경로를 연결하여 두 경로의 특징을 결합하는 것이 직관적이다. 이 설계는 멀티모달 딥 러닝 작업에서 널리 채택되었다. 그러나 단순히 벡터 연결만으로는 사용자와 항목 잠재 기능 사이의 상호작용을 설명하지 않으며, 이는 collaborative filtering 효과를 모델링하기에 불충분하다. 이 문제를 해결하기 위해 standard MLP를 사용하여 사용자와 항목 잠재 기능 사이의 상호 작용을 학습하여 연결된 벡터에 숨겨진 레이어를 추가할 것을 제안한다. 이러한 의미에서, 우리는 모델에 고정된 element-wise product을 사용하는 GMF의 방식이 아니라 pu와 qi 사이의 상호 작용을 학습하기 위해 많은 수준의 유연성과 비선형성을 제공할 수 있다. 보다 정확하게, 우리의 NCF 프레임워크 아래의 MLP 모델은 다음과 같이 정의된다.
$$
z_1=\phi_1(p_u,q_i)= \begin{bmatrix} p_u \\ q_i \end{bmatrix}, \\
\phi_2(z_1) = a_2(W^T_2z_1+b_2), \\
... \\
\phi_L(z_{L-1}) = a_L(W^T_Lz_{L-1}+b_L), \\
\hat{y}_{ui} = \sigma(h^T\phi_L(z_{L-1})), \\
 \\
 W_x, b_x, a_x  : \mbox{x-th layer’s perceptron의 weight matrix, bias vector, and activation function}
$$
MLP 레이어의 활성화 기능을 위해 sigmoid, tanh, ReLU를 자유롭게 선택할 수 있다. 각각 분석을 해보자면,

1) sigmoid는 각 뉴런이 (0,1) 안에 있는 것을 제한하고, 모델의 성능을 제한할 수 있으며, 뉴런의 출력이 0 또는 1에 가까울 때 뉴런이 학습을 중단하는 포화 상태를 겪는 것으로 알려져 있다.

2) hanh이 더 나은 선택이고 널리 채택되었음에도 불구하고, 그것은 시그모이드(tanh(x/2) = 2µ(x) - 1)의 축소된 버전으로 볼 수 있기 때문에 시그모이드 문제를 어느 정도 완화시킬 뿐이다.

3)이와 같이, 우리는 생물학적으로 더 그럴듯하고 포화 상태가 아닌 것으로 입증된 ReLU를 선택한다. 더욱이, 그것은 sparse activation를 장려하고 sparse 데이터에 적합하며 모델이 과적합할 가능성을 낮게 만든다. 우리의 경험적 결과는 ReLU가 tanh보다 약간 더 나은 성능을 제공하며, 이는 다시 sigmoid보다 훨씬 더 우수하다는 것을 보여준다. 네트워크 구조의 설계와 관련하여, 일반적인 해결책은 타워 패턴을 따르는 것인데, 바닥 층은 가장 넓고 각각의 연속적인 층은 더 적은 수의 뉴런을 가지고 있다. 그 전제는 더 높은 layers를 위해 적은 수의 hidden units을 사용함으로써 데이터의 추상적인 특징을 더 많이 배울 수 있다는 것이다. 우리는 탑 구조를 경험적으로 구현하여 연속적인 각 상위 계층에 대한 층 크기를 절반으로 줄인다.

### 3.4 Fusion of GMF and MLP

지금까지 NCF의 두 가지 인스턴스화, 즉latent feature interactions을 모델링하기 위해 linear kernel을 적용하는 GMF와 non-linear kernel 을 사용하여 데이터에서 상호 작용 함수를 학습하는 MLP를 개발했다. 그런 다음 문제가 발생한다: NCF 프레임워크에서 GMF와 MLP를 어떻게 융합하여 서로 강화하여 복잡한 사용자-항목 상호 작용을 더 잘 모델링할 수 있는가?

간단한 해결책은 GMF와 MLP가 동일한 임베딩 계층을 공유한 다음 상호 작용 함수의 출력을 결합하는 것이다. 이 방법은 잘 알려진 신경 텐서 네트워크(NTN)와 유사한 정신을 공유한다.

구체적으로, GMF와 one-layer MLP를 결합하는 모델은 다음과 같이 공식화될 수 있다.
$$
\hat{y}_{ui} = \sigma(h^Ta(p_u⊙q_i + W \begin{bmatrix} p_u \\ q_i \end{bmatrix} + b))
$$
그러나 GMF와 MLP의 임베딩을 공유하면 융합 모델의 성능을 제한할 수 있다. 예를 들어, 이는 GMF와 MLP가 동일한 크기의 임베딩을 사용해야 한다는 것을 암시한다. 두 모델의 최적의 임베딩 크기가 많이 변화하는 데이터 세트의 경우 이 솔루션은 최적의 앙상블을 얻지 못할 수 있다.

융합된 모델에 더 많은 유연성을 제공하기 위해, 우리는 GMF와 MLP가 별도의 임베딩을 학습할 수 있도록 허용하고, 마지막 숨겨진 계층을 연결하여 두 모델을 결합한다. 그림 3은 우리의 제안을 예시하며, 그 공식은 다음과 같다.
$$
\phi^{GMF} = p^G_u ⊙ q^G_i , \\
\phi^{MLP} = a_L(W^T_L(a_{L-1}(...a_2(W^T_2\begin{bmatrix} p^M_u \\ q^M_i \end{bmatrix} + b_2)..))+b_L), \\
\hat{y}_{ui} = \sigma(h^T\begin{bmatrix} \phi^{GMF} \\ \phi^{MLP} \end{bmatrix}) \\
p^G_u , p^M_u :  \mbox{user embedding for GMF and MLP} \\
q^G_i and q^M_i : \mbox{item embeddings for GMF and MLP}
$$
앞에서 논의한 바와 같이, 우리는 ReLU를 MLP 계층의 activation function으로 사용한다. 이 모델은 user–item latent structure를 모델링하기 위해 MF의 선형성과 DNN의 비선형성을 결합한다. 우리는 이 모델을 Neural Matrix Factorization의 줄임말인 "NeuMF"라고 한다. 각 모델 parameter에 대한 model의 파생 모델은 standard back-propagation를 사용하여 계산할 수 있으며, 공간 제한으로 인해 여기서 생략됩니다.

### *3.4.1 Pre-training*

NeuMF의 목적 함수의 non-convexity 때문에, 그레이디언트 기반 최적화 방법은  locally-optimal된 솔루션만 찾는다. initialization은 딥러닝 모델의 수렴과 성능에 중요한 역할을 하는 것으로 보고되었다. NeuMF는 GMF와 MLP의 앙상블이기 때문에, 우리는 GMF와 MLP의 사전 훈련된 모델을 사용하여 NeuMF를 초기화할 것을 제안한다. 우리는 먼저 수렴될 때까지 random initializations를 통해 GMF와 MLP를 훈련시킨다. 그런다음 우리는 각각 model parameters를 NeuMF’s parameters에 상응하는 부분에 initializations로 사용한다. 유일한  수정 부분는 output layer에서 두 모델의 weights를 연결하는 것이다.
$$
h←\begin{bmatrix} αh^{GMF} \\ (1-α)h^{MLP} \end{bmatrix}, \\
h^{GMF} , h^{MLP} : \mbox{pre-trained된 GMF,MLP의 h vector} \\
α : \mbox{사전 훈련된 두 모델 간의 절충을 결정하는 hyper-parameter.}
$$
GMF와 MLP를 처음부터 교육하기 위해, 우리는 빈번하지 않은 매개 변수에 대해 더 작은 업데이트를 수행하여 각 파라미터에 대한 learning rate를 조정하는 Adaptive Moment Assuration(Adam)을 채택한다. 아담 방법은 바닐라 SGD보다 두 모델에 대한 수렴 속도가 더 빠르며 학습 속도를 조정하는 고통을 덜어준다. 사전 훈련된 매개 변수를 NeuMF에 공급한 후, 우리는 아담이 아닌 바닐라 SGD로 최적화한다. 아담이 매개 변수를 제대로 업데이트하기 위해 모멘텀 정보를 저장해야 하기 때문이다. 사전 훈련된 모델 매개 변수만으로 NeuMF를 초기화하고 모멘텀 정보 저장을 포기하기 때문에 모멘텀 기반 방법을 사용하여 NeuMF를 더 최적화하는 것은 적절하지 않다.

