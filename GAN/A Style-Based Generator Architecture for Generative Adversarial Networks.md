# A Style-Based Generator Architecture for Generative Adversarial Networks

2019년 NVIDIA에서 발표한 논문.

## Abstract

  Style transfer 문헌을 차용하여 Generative adversarial networks를 위한 alternative generator architecture를 제안한다.

새로운 아키텍처는 자동으로 학습되고 비지도의 높은 수준의 속성(e.g., pose and identity when trained on human faces)과 생성된 이미지(e.g., 주근깨, 머리카락)의 stochastic variation을 separation하고 직관적으로 scale-specific control이 가능하게 한다.

새로운 generator는 기존 distribution quality metrics 측면에서 SOTA(*state-of-the-art*)를 향상시키고, 더 나은 interpolation properties을 입증하며 latent factor of variation을 더 잘 disentangled 하게 한다.

Interpolation quality와 disentanglement을 정량화(quantify) 하기 위해 모든 generator 아키텍처에 사용할 수 있는 두가지 새로운 자동화된 방법을 제안한다.

마지막으로, 새롭고 매우 다양한 고품질의 human faces 데이터셋을 소개한다.

## 1. Introduction

  Generative methods, 특히 GAN에 의해 생성된 이미지의 해상도와 퀄리티는 최근 빠르게 향상되고 있다.

그러나 아직 generators는 blackboxes로써 여겨지며, 이미지 합성 프로세스의 다양한 측면, 예를 들어 stochastic features의 기원에 대한 이해는 여전히 부족하다.

Latent space의 properties 또한 잘 이해되지 않았으며, 일반적으로 입증되는 latent space interpolations는 서로 다른 generators를 비교할 수 있는 정량적 방법을 제공하지 않는다.

  Style transfer literature에 의해 동기부여 받아, 이미지 합성 프로세스(image synthesis process)를 다루는 새로운 방법을 노출하는 방식으로 generator architecture를 재설계 한다.

우리의 generator는 학습된 contant input으로부터 시작하여 latent code를 기반으로 각 convolution layer에서 이미지의 "style"을 조정하여 서로 다른 스케일로 이미지 features의 강도를 직접 제어한다.

이러한 아키텍처 변화는 네트워크에 직접 유입되는 노이즈와 결합하여 생성된 이미지에서 높은 수준의 attributes(예: pose, identity)을 stochastic variation(예: 주근깨, 머리카락)로부터 자동으로 분리하고 직관적인 스케일별 mixing 및 interpolation operations을 가능하게 한다.

우리는 어떤 방법으로도 discriminator와 loss function을 수정하지 않고, 따라서 우리의 work는  GAN  loss functions, regularization 및 하이퍼 파라미터에 대한 지속적인 논의와 직교(?)한다.

  우리의 generator는 input latent code를 intermidiate latent space에 임베딩하고 있으며, 이는 네트워크에서 variation의 요인이 어떻게 표현되는지에 심오한 영향을 미친다.

Input latent space은 training data의 확률 밀도를 반드시 따라야 하며, 이로 인해 어느 정도 피할 수 없는 얽힘이 발생한다고 주장한다.

우리의 intermediate latent space는 제약이 없기 때문에 분리될 수 있다.(해방될 수 있다??)

Latent space disentanglement의 degree를 추정하기 위한 이전의 방법들은 우리의 경우에 직접적으로 적용할 수 없기 때문에, 위는 generator의 이러한 측면을 정량화하기 위해 perceptual path length와 linear separability라는 두가지 새로운 자동화 metrics를 제안한다. 

이러한 metrics를 사용해서, 우리는 기존의 generator architecture와 비교했을 때 우리의 generator가 서로다른 variation의 factor의 더 linear하고, 덜 entangled한 representation을 허용한 다는 것을 볼 수 있다.

  마지막으로, human faces (Flickr-Faces-HQ, FFHQ)의 새로운 데이터셋을 보여주는데, 이것은 더 높은 퀄리티를 제공하고 기존 고해상도 datasets보다 상당히 더 광범위한 변화를 다룬다.

우리는 이 데이터셋을 소스 코드 및 사전 교육된 네트워크와 함께 공개했다. 동반된 비디오는 밑에 링크에서 볼 수 있다.

https://github.com/NVlabs/stylegan

## 2. Style-based generator

  전통적으로 latent code는 input layer, 즉 feed-forward network의 첫 번째 레이어를 통해 generator에게 제공된다(Fig. 1a). *(기존의 generator (a)는 input latent vector (Z)가 직접 convolution, upsampling 등을 거쳐 이미지로 변환되는 구조이다.)*

![Screen Shot 2021-08-02 at 5.28.23 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-02 at 5.28.23 PM.png)

우리는 대신 입력 계층을 완전히 생략하고 학습된 상수에서 시작하여 이 설계에서 벗어난다(Fig. 1b 오른쪽). *(  **w**는 constant tensor가 이미지로 변환되는 과정에서 스타일을 입히는 역할을 수행함으로써 다양한 스타일의 이미지를 만들어낼 수 있다.)*

Latent code **z**가 input latent space Z에 주어졌을 때, non-linear mapping network f : Z → W 는 먼저 **w** ∈ W (Fig. 1b 왼쪽)을 생산한다. *(style-based generator (b) 의 경우, z가 fully-connected layer로 구성된 mapping network을 거쳐 intermediate latent vector (w)로 먼저 변환된다. )*

단순하게, 두 space 모두 dim을 512로 설정하고 mapping *f*는 섹션 4.1에서 분석할 8-layer MLP를 사용하여 구현된다.

학습된 affine tranformations를  synthesis network *g*의 각 convolution layer 후 adaptive instance normalization(AdaIN)을 제어하는 `y =(ys,yb)` *styles* 을 전문으로 다룬다.

AdalIN operation은 다음과 같다.
$$
AdalIN(x_i,y) = y_{s,i} \frac{x_i μ(x_i)}{\sigma(x_i)} + y_{b,i} 
$$
이때, feature map xi는 별도로 normalized된 다음, style **y** 의 해당 scalar compoments를 이용해 scaled 및 biaed된다.

따라서 y의 dimension은 해당 layer의 feature map 수의 두 배가 된다.

  우리의 approach를 style transfer와 비교했을 때, 우리는 example image 대신 벡터 **w**에서 invariant style **y**을 계산한다.