# Meta-Learning : Learning to Learn Fast
`Learning to Learn` 이라고 알려져 있는 Meta-Learning은 몇몇 training example을 통해 모델로 하여금 새로운 개념과 기술을 빠르게 학습시키는 것을 나타낸다.

## Meta-Learning Problem
 meta-learning model : 학습하는 task에 대한 다양성(varity)를 학습하고, 여러 tasks의 분포 상에서 최고의 성능을 내도록 최적화 

각 task는 dataset $\mathcal{D}$ 로 구성되어 있고, 여기에는 각각 feature vectors와 true labels가 포함되어 있다. 이 때 optimal model parameter는 다음과 같다.
$$
\theta^* = \argmin_\theta \mathbb{E}_{\mathcal{D}~p(\mathcal{D})}[\mathcal{L}_\theta(\mathcal{D})]
$$
>  여러 dataset 중 sampling된 dataset $\mathcal{D}$ 에 대해서 Loss function $\mathcal{L}_\theta(\mathcal{D})$를 최소화할 수 있는 $\theta$를 찾겠다는 의미.   
> 일반적인 learning task와 다른 점은 dataset 자체가 하나의 data sample로 활용되고 있다는 점이다.

Few-show Classification : Supervised learning 상에서 meta-learning을 활용한 예시   

dataset $\mathcal{D}$는  크게 두가지로 나뉜다.    
> dataset $\mathcal{D} = \begin{cases}
\text{support set}\ S, & \text{for learning} \\
\text{prediction set}\ B, & \text{for training \& testing}
\end{cases}$ 

따라서 dataset $\mathcal{D} = <S,B>$ 로 표현할 수 있다.

dataset $\mathcal{D}$는 여러 쌍의 feature vector와 label을 포함하고 있으므로 $\mathcal{D} = {(\mathbf{x_i},y_i)}$ 라 표현할 수 있는데, 이 때 각 label은 label set $\mathcal{L}$에 포함되어 있다.

Parameter $\theta$를 가진 Classifier $f_\theta$ : 주어진 데이터가 feature vector $\mathbf{x}$가 class $y$에 속할 확률 $P_\theta(y|\mathbf{x})$ 를 output

Optimal parameter는 dataset $\mathcal{D}$ 내에 있는 여러 training batch $B$에 대해 **true label을 얻을 수 있는 확률**을 높일 수 있어야 함.
$$
\theta^* = \argmax_\theta \mathbb{E}_{(\mathbf{x},y)\in\mathcal{D}}[P_\theta(y|\mathbf{x})] \\
\theta^* = \argmax_\theta \mathbb{E}_{B\sub\mathcal{D}}[\sum_{(\mathbf{x},y)\in B} P_\theta(y|\mathbf{x})] \qquad ; {\scriptstyle \text{trained with miti-batches}}
$$ 

Few-show classification의 목표 : `fast learning`을 위해 추가한 약간의 support set을 갖고 unknown label에 대한 데이터의 prediction error를 줄이는 것

dataset에 약간의 `fake`를 가해 모델이 모든 label에 대해 인지하고, optimization procedure를 수정하는 것을 막고, 궁극적으로 fast learning이 이루어질 수 있도록 하는 것이다.

> 1. Label set $\mathcal{L}$에서 일부 sampling. $L \sub \mathcal{L}$
> 2. dataset으로부터 support set 과 training batch sampling    
> $S^L \sub \mathcal{D}, B^L \sub \mathcal{D}$   
> 이 때 두 개의 set 모두 1.에서 sampling된 label set에 속하는 label을 가진 데이터만을 가지고 있어야 함   
> $y \in L, \forall(x,y) \in S^L, B^L$   
> 3. support set : model의 input   
> 4. Final optimization 단계 : mini-batch $B^L$을 이용해 loss를 계산하고 backpropagation을 이용하여 model parameter를 update한다.

이 때 sampling한 $(S^L, B^L)$을 하나의 data point로 고려하여 모델이 다른 dataset에 대해서도 generalize할 수 있도록 학습된다.    

빨간색으로 표시된 meta-learning 관련 term을 추가하면 다음과 같다.
$$
\theta = \argmax_\theta {\color{Red}E_{L \sub \mathcal{L}}[}E_{{\color{Red}S^L \sub \mathcal{D},} B^L \sub \mathcal{D}} [\sum _{(x,y) \in B^L} P_\theta(x,y{\color{Red},S^L}) ]{\color{Red}]}
$$

## Metric-Based Approach
Metric-Based meta-learning은 nearest neighbors algorithm과 비슷하다.
label $y$에 대한 predicted probability는 support set sample의 label에 대한 weighted sum과 같다. weight는 $k_\theta$를 통해 구할 수 있는데, 이 값은 두 개의 data sample간의 similarity 를 나타내는 것이다.
> 어렵다
$$ 
P_\theta(y|\mathbf{x},S) = \sum_{\mathbf{x}_i,y_i\in S} k_\theta(\mathbf{x},\mathbf{x}_i)y_i
$$

Metric-based meta-learning model이 잘 동작하기 위해서는 좋은 kernel function을 학습하는 것이 중요하다.

아래는 Input data에 대한 embedding vector를 학습하고 이를 통해 적절한 kernel function을 설계하는 방법을 소개한다.

### Convolutional Siamese Neural Network






--- 
참고 자료   
https://talkingaboutme.tistory.com/entry/DL-Meta-Learning-Learning-to-Learn-Fast