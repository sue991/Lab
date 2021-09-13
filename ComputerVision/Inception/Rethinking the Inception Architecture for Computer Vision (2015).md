# Rethinking the Inception Architecture for Computer Vision (2015)

## Abstract

  Convolutional networks는 다양한 분야의 최신 CV 솔루션의 핵심이다. 2014년부터 very deep convolutional network가 주류를 이뤘으며, 다양한 벤치마크에서 실질적인 성능 이득을 얻었다.

비록 모델의 크기나 계산 비용이 증가하긴 하지만, 충분한 양의 학습 데이터만 제공된다면 대부분의 작업에서 즉각적인 성능 향상이 이루어진다. 

 또한 convolution의 계산적인 효율성이나 적은 수의 parameter를 사용한다는 특징으로 인해, mobile vision이나 big-data scenario와 같은 다양한 케이스에 적용 가능하게 한다.

이 논문에서는 다음의 두 방법을 통해, 네트워크의 크기를 효율적으로 키우는 방법을 탐색한다.

1. **Suitably factorized convolutions**
2. **Aggressive regularization**

제안하는 방법을 ILSVRC 2012 classification challenge의 validation set에 테스트하여, state-of-the-art 기법에 비해 상당한 성능 향상을 보였다. Inference 한 번에 소요되는 계산 비용이 [ less than 25 million paramters / 5 billion multiply-adds ]인 네트워크를 가지고 테스트 한 결과는 다음과 같다.

- Single frame evaluation에서 top-1 error가 21.2%이고, top-5 error가 5.6%
- 4가지 모델을 ensemble한 multi-crop evaluation에서 top-1 error가 17.3%이고, top-5 error가 3.5%

## Introduction

 2012년 ImageNet competition에서 우승한 **"AlexNet"** 은  이후 다양한 종류의 컴퓨터 비전 분야에 성공적으로 적용됐다. 예를들어, Object-Detection, Segmentation, Human pose estimation, Video classification, Object tracking, superresolution 등..

이러한 성공은 더 높은 CNN의 성능에 초점을 맞추는 연구들의 방향을 촉발했으며, 2014년 이후 너 깊고 넓은 네트워크를 활용하여 network achitecture의 퀄리티가 매우 향상되었다.

VGGNet과 GoogLeNet은 ILSVRC 2014에서 비슷한 성과를 나타냈으며, 이로부터 classification 성능의 향상이 다양한 응용 분야에서의 상당한 성능 향상으로 이어지는 경향이 있음을 알 수 있었다.

이것은 CNN 구조의 개선으로부터 visual features에 의존하는 대부분의 computer vision 분야에서의 성능 향상이 이루어질 수 있음을 의미한다. 

또한 네트워크 퀄리티의 향상은 Detection의 region proposal의 경우와 같이, [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)의 성능이 hand-engineered solution만큼 성능이 나오지 않았던 분야에서도 CNN을 활용할 만한 상황을 만들었다. (네트워크의 퀄리티가 높아지면서 CNN을 위한 새로운 application domain이 생겼다.)

  VGGNet은 간단한 구조를 가지고 있다는 장점이 있지만, cost가 높다(네트워크 평가 계산 비용이 높다).

반면, GoogLeNet의 Inception architecture는 memory나 computational budget에 대해 엄격한 제약 조건이 주어지는 경우에서도 잘 수행되도록 설계되어 있다.

> 예를 들어, AlexNet은 약 60 million개, VGGNet은 이에 약 3배 정도 되는 parameter를 사용하는 반면, GoogleNet은 오직 500만개의 parameters를 사용한다.

  Inception의 계산 비용은 VGGNet이나 혹은 더 좋은 성능을 달성한 후속 네트워크보다 훨씬 적다.

이는 mobile vision 환경과 같이 메모리나 computational capacity가 제한된 환경에서 합리적인 비용으로 big-data를 다루는 경우에 Inception networks를 활용하기에 좋음을 보여준다.

물론 memory usage에 특화된 solution을 사용하거나 computational tricks으로 특정 동작의 수행을 최적화하는 등의 방법으로 문제를 어느정도 완화시킬 수 있지만, 이는 계산 복잡성이 가중될 뿐만 아니라, 이와 같은 기법들이 inception 구조의 최적화에도 적용된다면 효율성의 격차가 다시 벌어질 것이다.

 Inception구조의 복잡성은 네트워크의 변경을 더욱 어렵게 만든다. 만약 구조를 단순히 확장한다면, 계산적인 장점의많은 부분이 손실될 수 있다. 또한, GoogleNet에서는 구조적인 결정들에 대한 디자인 철학을 명확하게 설명하지 않아, 효율성을 유지하면서 새로운 use case에 적용하는 등의 구조 변경이 훨씬 어려워진다.

예를 들어 만약 어떤 Inception-Style model의 capacity를 올릴 필요가 있다고 생각한다면, 모든 filter bank의 크기를 2배로 늘리는 간단한 방법을 취할 수 있지만, 이것은 계산 비용과 파라미터의 수를 4배로 늘리게 되고, 이것이 관련된 많은 이점이 없을 경우, 대부분의 real-world 시나리오에서 사용할 수 없거나 불합리한 것으로 판명될 수 있다.

이 논문에서는 CNN의 효율적인 확장에 유용한 몇 가지 일반적인 원칙과 최적화 아이디어를 설명하는 것으로 시작한다. 

우리의 원칙들은 inception-type에 제한되어있는 원칙이 아니며, 일반적으로 inception style building block 구조가 이러한 제약들을 자연스럽게 통합하여 이들의 효과를 관찰하기 더 쉽다.

이는 근처 구성요소의 구조적인 변화에 따른 영향을 완화해주는 dimensional reduction과 inception의 병렬 구조를 충분히 사용함으로써 가능하다.

여전히 모델의 높은 성능을 유지하기 위해서는 몇 가지 지침 원칙을 준수할 필요가 있으니 주의할 필요가 있다.

## General Design Principles

  여기서는 large-scale experimentation에 근거하여 CNN의 다양한 구조적적인 선택에 대한 몇몇 디자인 원칙을 설명한다.

아래에 있는 원칙들의 활용은 이들의 정확성이나 타당성 평가를 위해선 추가적인 실험적 증거가 필요할 것이다.

>  이러한 원칙들에서 크게 벗어나면 네트워크의 성능이 저하되는 경향이 있으며, 해당 부분을 수정하면 일반적으로 구조가 개선된다.

### 1. Avoid representational bottlenecks

Feed-forward networks는 input layer로부터 classifier나 regressor에 이르는 비순환 그래프로 나타낼 수 있으며, 정보가 흐르는 방향을 명확하게 알 수 있다.

Input에서 output까지의 모든 layer의 경우, layer를 통과하는 정보의 양에 접근할 수 있다. 이 때, **극단적인 압축으로 인한 정보의 bottleneck 현상이 발생하지 않도록 해야한다**.

> 특히, 네트워크의 초반부에서 일어나는 bottleneck을 주의할 필요가 있다. 결국에는 모든 정보의 출처가 입력 데이터인데, 초반에서부터 bottleneck이 일어나서 정보 손실이 발생한다면 아무리 네트워크가 깊어진다 한들, 정보의 원천이 부실해지므로 성능의 한계가 발생하기 때문으로 생각된다.

일반적으로 representation size는 final representation에 도달하기 전 input에서 output으로 갈 수 록 서서히 감소해야 한다.

> *Representation은 각 layer의 출력으로 생각하면 된다. 일반적으로 pooling 과정을 통해 feature map size가 작아지는데, 이 과정의 필요성을 말하는 것으로 보인다.*

이론적으로, correlation structure와 같은 중요한 요소를 버리기 때문에, 정보를 dimensionality of representation으로만 평가할 수 없다.

### 2. Higher dimensional representations

높은 dimensional representation은 network 내에서 locally하게 처리하기 더 쉽다.

CNN에서 activations per tile을 늘리면 disentangled feature를 많이 얻을 수 있으며, 네트워크가 더 빨리 학습하게 될 것이다.

> *Conv layer의 filter map개수를 늘리면, 다양한 경우의 activated feature map을 탐지할 수 있고, 이를 통해 네트워크의 학습이 빨라질 수 있다는 뜻으로 보인다.*

### 3. Spatial aggregation

Spatial aggregation은 representational power에 크거나 작은 loss 없이 낮은 차원의 embeddings에서 수행될 수 있다.

예를 들어, 보다 광범위한 convolution을 수행하기 전에, 심각한 부작용 없이 spatial aggregation 이전에 input representation의 dimension을 줄일 수 있다. 

그 이유는 output이 spatial aggregation context에서 사용되는 경우, 인접한 유닛 간의 강한 correlation이 dimension reduction동안 정보의 손실을 훨씬 적게 유발하기 때문이라고 가정한다.

이러한 signal은 쉽게 압축 할 수 있어야한다는 점을 감안하면, dimension reduction으로 인해 학습 속도가 빨라진다.

>Convolution 연산을 spatial aggregation이라 표현하는 것으로 보인다. Signal의 압축은, 학습 과정에서 네트워크의 각 layer를 거쳐가며, 원하는 동작을 위한 판단에 필요한 feature를 입력 데이터로부터 추출하는 작업을 signal의 압축 과정으로 생각한다면 쉽게 이해할 수 있다. 즉, **convolution을 다수 수행하는 경우에는 적절한 dimension reduction을 해주는 것이 빠른 학습에 도움 된다**는 것으로 보면 된다.
>
>이는 **출력이 spatial aggregation에 사용되는 경우, 인접한 unit 간의 강력한 상관 관계로 인해 dimension reduction 중의 정보 손실이 훨씬 줄어들 것이라는 가설**에 근거한 원칙이다.

### 4. Balance the width and depth of the network

Network의 최적의 성능은 각 stage의 filter 수와 네트워크의 depth의 밸런스를 조정하여 달성할 수 있다.

Network의 width와 depth를 증카시키는 것은 더 높은 퀄리티의 네트워트에 기여할 수 있다.

그러나 둘 다 병렬로 증가하면 일정한 양의 계산에 대한 최적의 향상에 도달할 수 있다.

> *즉,* **늘릴 때 늘리더라도, computational budget이 depth와 width 간에 균형 잡힌 방식으로 할당되도록 네트워크를 구성***해야 최적의 성능을 보일 것이다.*

비록 이러한 원칙들이 타당하더라도, 이를 활용하여 네트워크의 성능을 향상시키는 것은 간단하지 않다. 따라서, 모호한 상황에서만 이 아이디어들을 고려하도록 하자.

## Factorizing Convolutions with Large Filter Size

GoogLeNet network의 이점 중 대부분은 dimension reduction을 충분히 사용하면서 발생한 것이다. 이는 계산 효율이 좋은 방식으로 convolution을 factorizing하는 것의 특별한 케이스로 볼 수 있다.

1x1 conv layer 다음에 3x3 conv layer가 오는 경우를 생각해보자. Vision network에서, 인접한 activation의 output은 매우 높은 상관관계가 예상된다. 따라서 aggregation 이전에 activation이 줄어들 수 있으며, 유사하게 표현되는 local representation을 가질 수 있다.

> *상관 관계가 높은 activation 간에는 유사한 표현력을 지니며, 이들의 수가 줄어들더라도 상관없는 것으로 생각된다.*

  여기서는 다양한, 특히 모델의 계산 효율을 높이기 위한 세팅에서 factorizing convolutions의 방법들을 설명한다.

Inception networks는 fully convolutional하기 때문에, 각 weight는 activation 당 하나의 multiplication에 해당한다. 따라서, 계산 비용을 줄이면 parameter 수가 줄어들게 된다.

이는 적절한 factorizing이 이루어지면, 더 많은 disentangled parameter를 얻을 수 있으며, 이에 따라 빠른 학습이 가능하다는 것을 의미한다.

또한 컴퓨팅 및 메모리 절감 효과를 사용하여 네트워크의 filter-bank size를 늘리는 동시에 각 모델 복제본을 단일 컴퓨터에서 교육할 수 있다.

### 1. Factorization into smaller convolutions

더 큰 spatial filters(e.g. 5 x 5 or 7 x 7)를 갖는 Convolution은 computation 측면에서 불균형하게 비싼 경향이 있다.

  예를 들어 n개의 filter로 이루어진 5x5 convolution 연산의 경우, 같은 수의 filter를 사용하는 3x3의 convolution보다 계산 비용이 259259로, 약 2.78배 더 비싸다.

물론 5x5 filter는 이전 layer에서 머리 떨어진 unit의 activation간의 signal 간 dependencies를 포착할 수 있기 때문에 filter의 크기를 줄이는 데 그만큼 표현력을 위한 비용이 커지게 된다.

그러나, 우리는 5x5 convolution을 동일한 input size와 output depth를 가지면서, 더 적은 parameter를 가진 multi-layer 네트워크로 대체할 방법에 대해 고민한다.

![Screen Shot 2021-08-25 at 10.07.04 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-25 at 10.07.04 PM.png)

5x5 convolution의 computation graph를 확대한다면, 각 출력은 입력에 대해 5x5 filter가 sliding하는 형태의 소규모 fully-connected 네트워크처럼 보인다. (Figure 1)

여기선 vision network를 구축하고 있기 때문에, fully-connected component를 2-layer convolution로 대체하여 translation invariance을 다시 이용하는 것이 자연스러워 보인다.

> 즉 첫 번째 레이어는 3x3 conv이고, 두 번째 레이어는 첫 번째 레이어의 3x3 output grid 위에 연결된 fully connected layer이다 (Figure 1) . 이와 같이 input activation grid에 sliding하는 filter를 5x5 conv에서 2-layer 3x3 conv로 대체하는 것이 여기서 제안하는 factorization 하는 방법이다. (Fig 4, 5와 비교)

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-25 at 10.14.12 PM.png" alt="Screen Shot 2021-08-25 at 10.14.12 PM" style="zoom:50%;" />     <img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-25 at 10.14.31 PM.png" alt="Screen Shot 2021-08-25 at 10.14.31 PM" style="zoom:50%;" />  

이 구조는 인접한 unit 간의 weight를 공유함으로써 parameter 수를 확실히 줄여준다.

절감되는 계산 비용을 예측 분석하기 위해, 일반적인 상황에 적용 할 수 있는 몇 가지 단순한 가정을 해보자. 우선 n=αm로 가정한다. 즉, activation이나 unit의 개수를 상수 α에 따라 결정한다.

> *5x5 convolution을 수행하는 경우엔* α*가 일반적으로 1보다 약간 크며, GoogLeNet의 경우엔 약 1.5를 사용했다.*

5x5 conv layer를 2-layer로 바꾸는 경우, 두 단계로 확장하는 것이 합리적이다. 여기선 문제를 단순화 하기 위해 확장을 하지 않는 α=1을 고려한다.

2-layer의 경우, 각 단계에서 filter 수를 √α만큼 증가시키는 방법을 취할 수 있다.

만약 인접한 grid tile 간의 계산 결과를 재사용하지 않으면서 단순히 5x5 convolution sliding만 하게 된다면, 계산 비용이 증가한다. 이 때, 5x5 convolution sliding을 인접한 tile 간의 activation을 재사용하는 형태의 2-layer 3x3 convolution으로 나타낼 수 있으며, 이 경우에는 (9+9)/25 = 0.92로 계산량이 감소하게 된다. 즉, actorizing을 통해 28%의 상대적 이득을 얻는 것에 해당한다.

이 경우에도 parameter들은 각 unit의 activation 계산에서 정확히 한 번씩만 사용되므로, parameter 개수에 대해서도 정확히 동일한 절약이 일어난다.

여전히 두가지 일반적인 의문점이 생길 수 있다:

1. 위와 같은 replacement로부터 표현력 손실이 발생하는가?
2. 계산의 linear part에 대한 factorizing이 목적인 경우엔, 2-layer 중 first layer에서는 linear activation을 유지해야 하는가?

이에 대한 여러 실험을 수행(Figure 2)하여 factorization에 linear activation을 사용하는 것이 모든 단계에서 ReLU를 사용하는 것보다 성능이 좋지 않은 것을 확인했다고 한다. 

![Screen Shot 2021-08-25 at 10.40.08 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-25 at 10.40.08 PM.png)

우리는 이러한 이득들이 네트워크에서 학습할 수 있는 space of variation을 확대해준다고 보고, 특히 output activation을 batch-normalize할 경우 이러한 향상이 강하게 나타난다. Dimension reduction components에 linear activation을 사용하는 경우에도 유사한 효과를 볼 수 있다.

> *네트워크가 학습할 수 있는 space of variation은 모델의 capacity를 말한다.*

### 2. Spatial Factorization into Asymmetric Convolutions

위의 결과에 따르면, filter의 크기가 3x3보다 큰 convolution은 항상 3×3 convolution의 sequence로 축소될 수 있으므로 이를 이용하는 것은 보통 효율적이지 않다.

여전히 2x2 convolution과 같이 더 작게 factorize할 수 있긴 하지만, n x 1과 같은 asymmetric convolutions를 사용하는 것이 2 x 2보다 더 좋을 수 있다는 것이 밝혀졌다.

예를들어 3x1 convolution 뒤에 1x3 convolution을 사용한 2-layer를 sliding 하는 것과, 3×3 convolution의 receptive field는 동일하다. (Fig. 3)

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-26 at 3.55.25 PM.png" alt="Screen Shot 2021-08-26 at 3.55.25 PM" style="zoom:50%;" />



여전히 input filter 수와 output filter 수가 같은 경우, two-layer solution은 동일한 수의 output filter에 대해 33% 연산량이 싸다.

그에 비해, 3 × 3 convolution을 2 × 2 convolution으로 factorizing하는 것은 계산의 11% 절감에 불과하다.

  이론적으로, 어느 n x n convolution이든 1×n 뒤에 n×1 convolution이 오는 형태로 대체할 수 있으며, computational cost 절약은  n이 커짐에 따라 극적으로 증가한다(Fig. 6).

![Screen Shot 2021-08-26 at 4.00.30 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-26 at 4.00.30 PM.png)

실제로, 우리는 초기 layer에서는 이 factorization를 채택하는 것이 잘 작동하지 않지만, medium grid-sizes에서는 매우 좋은 결과를 보여준다. (m × m feature map,  m 범위는 12와 20 사이)

그 수준에서, 1 × 7 컨볼루션과 7 × 1 컨볼루션으로 매우 좋은 결과를 얻을 수 있었다.

## Utility of Auxiliary Classifiers

GoogLeNet은 매우 깊은 네트워크의 수렴(convergence)을 개선시키기 위해 보조 분류기(Auxiliary Classifier)를 도입했다.

Original motivation은 매우 깊은 네트워크의 vanishing gradient를 해결하여 useful gradients 하 layer로 밀어넣어 즉시 useful하게 만들고, training 중의 수렴(convergence)를 개선는 것이었다.

Lee et al[11] 또한 auxiliary classifiers가 더 안정적인 학습과 더 나은 수렴을 촉진한다고 주장했다.

흥미롭게도, 우리는 auxiliary classifiers가 수렴을 개선하지 않는다는 것을 발견했다: 높은 정확도에 도달하기 전까지의 학습 과정에서는 auxiliary classifiers의 유무랑 관계없이 유사한 성능을 보였지만, 학습이 끝날 무렵에는 보조 분류기가 있는 네트워크에서 정확도를 앞지르기 시작하다가 결과적으론 조금 더 높은 성능에 도달하며 학습이 종료됐다고 한다.

또한 GoogLeNet에서는 두 개의 보조 분류기가 각각 다른 stage에 사용됐지만, 하위 auxiliary branch의 제거는 최종 성능의 악영향을 끼치지 않았다. 

이는 앞 단락의 이전 관측치와 함께 이 branch가 low-level feature를 진화시키는 데 도움이 된다는 가설(`"보조 분류기가 low-level feature의 발전에 도움이 된다"`)이 잘못되었을 가능성이 크다는 것을 의미한다. 대신에, 우리는 auxiliary classifiers가 regularizer로 동작한다고 주장한다.

이는 보조 분류기에서 batch-norm이나 dropout layer가 사용되는 경우 main classifier의 결과가 더 좋아진다는 사실에 근거가 된다. 이는 또한 batch normalization이 regularizer의 역할을 한다는 추측에 미약한 증거가 된다.

## Efficient Grid Size Reduction

전통적으로 convolution networks는 pooling operation을 사용하여 feature map의 grid size를 줄였다. Representational bottleneck을 피하기 위해, max 또는 avg pooling 을 적용하기 전에 네트워크 filters의 activation dimension이 확장된다.

> 예를 들어, d x d grid의 k개의 filter로 시작해서, d/2 × d/2 grid의 2k개의 filter에 도달하려면, 먼저 stride-1 convolution으로 계산을 하고 추가적인 Pooling을 적용한다.
>
> *Grid size가 줄어들기만 하는건 grid에 들어있던 정보를 보다 저차원의 데이터로 압축하는 것이기 때문에, 이를 병목 현상(bottleneck)으로 볼 수 있다. 이 때문에 filter의 개수를 먼저 늘려준다면 정보의 병목 현상을 완화시키는 효과가 있는 것이다.*

이것은 전반적인 계산 비용이 pooling 이전의 확장 단계에서 일어나는 `2d^2k^2` operation 에 크게 좌우된다는 것을 의미한다.

만약 pooling과 convolution의 위치를 바꾼다면, 계산 비용이 1/4로 감소된 `2(d/2)/^2k^2` 이 된다. 하지만, 이는 전반적인 representation의 차원이 (d/2)^2k로 낮아져서 표현력이 떨어지게 되어representational bottlenecks을 야기한다(Fig. 9)

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-26 at 5.49.09 PM.png" alt="Screen Shot 2021-08-26 at 5.49.09 PM" style="zoom:50%;" />

대신에 우리는 representational bottleneck을 피하면서 계산 비용도 줄일 수 있는 또다른 구조를 제안한다(Fig 10).

![Screen Shot 2021-08-26 at 5.51.27 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-26 at 5.51.27 PM.png)

 제안하는 방법은 stride가 2인 block 2개(P, C)를 병렬로 사용한다. 각 블록은 pooling layer와 conv layer로 이루어져 있으며, pooling은 maximum 혹은 average를 사용한다. 두 block의 filter bank는 Fig.10에 나타난 것처럼 concatenate로 연결 된다.

## Inception-v2

위에서 언급한 것들을 결합하여, 새로운 아키텍처를 제안한다.  이 구조는 ILSVRC 2012 classification benchmark에서 향상된 성능을 보였다. Network의 layout은 table 1에 나와있다.

![Screen Shot 2021-08-26 at 5.55.47 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-26 at 5.55.47 PM.png)

Section 3.1내용을 기반으로 전통적인 7 x 7 convolution을 3개의 3×3 convolution으로 factorizing했다.

Network의 Inception 부분의 경우,각각  35 x 35인 288개의 filters가 있는 기존 inception modules을 가지고 있다.

이는 Section 5의 내용을 기반으로 grid reduction 기술을 사용하여 768개의 filters를 가진 17x17 grid로 축소된다.

다음은 3.2절의 asymmetric fatorizing 기법을 이용한 inception module이 5개 뒤따르며, 마찬가지로 5장의 reduction 기법에 의해 grid가 8×8×1280으로 축소된다.

Grid size가 8×8로 가장 축소 된 단계에서는 Fig.6의 inception module이 2개 뒤따른다. 각 tile에 대한 출력의 filter bank size는 2048이 된다.

Inception module 내에 filter bank size를 포함한 네트워크의 자세한 구조는 본 논문의 tar 파일에 포함된 model.txt에 나와 있다. (어딨음?)

그러나 우리는 Section 2의 원칙들을 지키면 구조적 변화에도 비교적 안정적인 성능을 보인다는 것을 관찰했다. 비우리의 network는 42 layers만큼 깊음에도 불구하고 계산 비용은 GoogLeNet(22-layer)다 오직 2.5배만 높으며, VGGNet보다 훨씬 효율적이다.

## Model Regularization via Label Smoothing

  여기서는 training 중 label-dropout의 marginalized effect를 추정하여, classifier layer를 regularize하는 메커니즘을 제안한다.

  각 training example x에 대해, 모델은 각 label k ∈ {1 . . . K}에 대한 확률을 계산한다:
$$
p(k|x) = \frac{\exp(z_k)}{\sum^K_{i=1}\exp(z_i)}
$$
 zi는  `logits`  또는 `unnormalized log-probability`이다.

이 training example의 labels `q(k|x)`에 대한 ground-truth distribution 을 고려하여 normalized하면, 
$$
\sum_k q(k|x) = 1
$$
이 된다.

편의 상 example x에서 p and q를 독립적인 것으로 생각하자. 학습 데이터의 loss는 cross entropy로 정의된다.
$$
l = -\sum ^K_{k=1} \log(p(k))q(k)
$$
이것을 최소화하는 것은 예상된 label의 log-likelihood를 최대화 하는 것과 같다. 이때 label은 ground-truth distribution q(k)에 따라 선택된다.

Cross-entropy loss는 logins z_k에 대해 구별할 수 있으므로 deep models의 gradient training에 사용할 수 있다.

Gradient 다음과 같은 간단한 폼을 따른다:
$$
\frac{∂ℓ}{∂z_k} = p(k) - q(k), \\
\mbox{bounded in [-1,1]}
$$
  모든 k != y에 대해 q(y) = 1 및 q(k) = 0인 single ground-truth label y의 경우를 가정하자.

이 경우에, cross entropy를 최소화하는 것은 correct label의 log-likelihood를 최대화하는것과 같다.

Label y와 특정 예 x에서, log-likelihood는 q(k) = δk,y에서 최대화된다. 이때 δk,y는 Dirac delta인데, k = y 일때는 1이고 나머지 일때는 0이다.

유한한 z_k에 대해 maximum을 달성 할 수는 없지만, zy≫zk,∀k≠y인 경우에는 maximum에 근접할 수 있다.

> *즉, ground-truth label에 해당하는 logit이, 나머지 모든 logit들보다 훨씬 큰 경우에는 maximum log-likelihood에 근접할 수 있다.*

그러나, 이 경우 두 문제가 생길 수 있다

1. overfitting이 생길 수 있다 : 만약 모델이 각 학습 데이터를 ground-truth label에 모든 확률을 할당하도록 학습한다면, 일반화 성능을 보장할 수 없다.

2. Largest logit과 나머지 logit 간의 차이가 매우 커지도록 유도된다.

   > 이 특성이 [−1,1]의 값인 bounded gradient ∂ℓ/∂z_k 와 함께 쓰이게 되면 모델의 적응력을 감소시킨다.

위 문제들의 원인을 직관적으로 유추해보면, 모델이 prediction에 대한 confidence를 너무 높게 갖기 때문에 발생하는 것으로 볼 수 있다.





