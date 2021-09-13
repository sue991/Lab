# Going Deeper with convolutions(2014)

GoogLeNet(Inception v1) 이라고 불린다.

2014년 이미지 인식 대회(IRSVRC) 에 VGG를 간발의 차이로 이겨 1등을 차지한 모델이다.

GoogLeNet은 VGG19보다 좀 더 깊고, AlexNet(2012)보다 적은 파라미터를 사용한다.

딥러닝은 망이 깊을수록(deep), 레이어가 넓을수록(wide) 성능이 좋지만, 현실적으로 overfitting, vanishing 등의 문제로 큰 모델의 학습이 어렵다.

> GoogLeNet은 **22층**으로 구성되어 있고, AlexNet보다 12x 적은 파라미터 수(약 6.8M개)를 사용한다.

이러한 모델을 구현하려면 다음과 같은 현실적인 문제들이 존재한다.

> Network가 Sparse 할수록 좋은 성능을 낸다.
> *Arora et al* 에 나와있는 main result에 따르면, data set의 probability distribution이 매우 크고(large) sparse한 DNN으로 표현될 수 있다면, 높은 correlation을 가지는 output들과 이 때 활성화되는 망내 노드들의 클러스터들의 관계를 분석하여 layer별로 최적의 network topology를 구성할 수 있다고 한다.

그런데 실제 computing infrastructures에 있어서는 연산에 사용되는 matrix가 Dense해야 리소스 손실이 적다.

> 사용되는 데이터가 uniform distribution을 가져야 리소스 손실이 적어진다.

따라서 여기서는 *Arora et al*에 나와있는 내용을 기반으로 망 내 연결을 줄이면서(sparsity), 세부적인 matrix 연산에서는 dense한 연산을 처리하도록 구성하였다.

## Inception v1.

![Screen Shot 2021-09-03 at 2.17.49 PM](/Users/sua/Desktop/Screen Shot 2021-09-03 at 2.17.49 PM.png)

Conv 레이어를 앞서 설명한대로 sparse 하게 연결하면서 행렬 연산 자체는 dense 하게 처리하는 모델로 구성하였다.

(a)그림을 보면 일반적인 convolution 연산이다.

- 보통은 하나의 Convolution 필터로 진행을 하는데, 여기서는 작은 Conv 레이어 여러 개를 한 층에서 구성하는 형태를 취한다.

## 1x1 Convolution

Conv 연산은 보통 3차원 데이터를 사용하는데 여기에 batch_size를 추가하여 4차원 데이터로 표기한다. `(ex) : [B,W,H,C]`

보통 Conv 연산을 통해 WW, HH의 크기는 줄이고 CC는 늘리는 형태를 취하게 되는데,

- W, H는 Max-Pooling 을 통해 줄인다.
- C는 Conv Filter 에서 지정할 수 있다. 보통의 Conv는 C를 늘리는 방향으로 진행된다.
- 이 때 1x1 연산은 Conv 연산에 사용하는 필터를 1x1 로 하고 C는 늘리는 것이 아니라 크기를 **줄이는** 역할을 수행하도록 한다.
- 이렇게 하면 C 단위로 fully-conntected 연산을 하여 차원을 줄이는 효과(압축)를 얻을 수 있다. 이게 NIN. (Network in Network)

즉, 1x1 Conv는 **특성맵의 갯수를 줄이는 목적으로 사용되는 것이다.** 특성맵의 갯수가 줄어들면 그만큼 연산량이 줄어든다.

GoogLeNet에 사용된 모듈은 1x1 컨볼루션이 포함된 (b) 모델이다.

노란색 블럭으로 표현된 1x1 Conv를 제외한 나이브(naïve) 버전을 살펴보면, 이전 layer에서 생성된 feature map을 1x1 Conv, 3x3 Conv, 5x5 Conv, 3x3 MaxPooling 하여 얻은 feature map들을 모두 함께 쌓아준다. 따라서 **좀 더 다양한 종류의 특성이 도출된다**. 여기에 1x1 컨볼루션이 포함되었으니 당연히 연산량은 많이 줄어들었을 것이다. 

>Max-Pooling 의 경우 1x1 이 뒤에 있는 이유는 MaxPooling은 output C를 조절할 수 없기 때문에 사이즈를 맞추기 위함이다.

![Screen Shot 2021-09-03 at 3.32.05 PM](/Users/sua/Desktop/Screen Shot 2021-09-03 at 3.32.05 PM.png)

Conv 연산을 좀 더 작은 형태의 Conv 연산 조합으로 쪼갤 수 있다. 이렇게 하면 정확도는 올리고, 컴퓨팅 작업량은 줄일 수 있다.

위의 표는 Inception 전체 Layer에서 사용되는 Conv 의 크기를 순서대로 나열한 표이다.

입력 이미지 크기는 `224x224x3` 이다.

 레이어 초반에는 인셉션 모듈이 들어가지 않는다. (효과가 없단다.)

`reduce` 라고 되어 있는 값은 앞단 1x1 Conv 의 C (channel) 값을 의미한다.

전체 GooLeNet의 구조는 다음과 같다.

<img src="/Users/sua/Desktop/Screen Shot 2021-09-03 at 3.35.48 PM.png" alt="Screen Shot 2021-09-03 at 3.35.48 PM" style="zoom:50%;" />

## Global Average Pooling

다른 모델(AlexNet, ResNet 등)에서는 모델 후반부에 fully connected (FC)레이어들이 연결되어 있다. 그러나 GoogLeNet은 FC 대신 **global average pooling**이라는 방식을 사용한다. global average pooling은 전 layer에서 산출된 feature map들을 각각 평균낸 것을 이어서 1차원 벡터를 만들어주는 것이다. 1차원 벡터를 만들어줘야 최종적으로 이미지 분류를 위한 softmax 층을 연결해줄 수 있기 때문이다. 만약 전 층에서 1024장의 7 x 7의 특성맵이 생성되었다면, 1024장의 7 x 7 feature map 각각 평균내주어 얻은 1024개의 값을 하나의 벡터로 연결해주는 것이다.

이렇게 해줌으로 얻을 수 있는 장점은 가중치의 갯수를 상당히 많이 없애준다는 것이다. 만약 FC 방식을 사용한다면 훈련이 필요한 가중치의 갯수가 7 x 7 x 1024 x 1024 = 51.3M이지만 **global average pooling을 사용하면 가중치가 단 한개도 필요하지 않다.** 

## Auxiliary classifier

  네트워크의 깊이가 깊어지면 깊어질수록 vanishing gradient 문제를 피하기 어려워진다. 

이 문제를 극복하기 위해서 GoogLeNet에서는 네트워크 중간에 두 개의 보조 분류기(auxiliary classifier)를 달아주어  중간 층에서도 Backprop을 수행하여 weight 갱신을 시도하였다.

> 따라서 총 3개의 softmax가 있다.

이 보조 분류기들은 훈련시에만 활용되고 사용할 때는 제거해준다. 

전체 Loss 값에 0.3 비율로 포함된다.

테스트 단계에서는 마지막 *softmax* 레이어만 실제로 사용한다.