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

$$
sdfsdkfjshdfgiuwrbjv
$$


$$
sdfsdfsdf
