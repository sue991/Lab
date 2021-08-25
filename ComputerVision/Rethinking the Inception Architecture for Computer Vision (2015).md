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

