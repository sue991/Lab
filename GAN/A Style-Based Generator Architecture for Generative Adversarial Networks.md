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

우리느 유사한 네트워크 아키텍처가 feedforward style transfer, unsupervised image-to-image translation, domain mixtures에 사용되고 있기 때문에 y에 대해 "style"이라는 단어를 재사용하기로 선택했다.

더 일반적인 feature transform과 비교했을 때, AdaIN은 효율성과 간결한 표현 때문에 특히 우리 목적에 잘 맞는다.

  마지막으로, explicit *noise inputs*을 도입하여 tochastic detail을 생성하기 위한 직접적인 수단으 generator에 제공한다.

이는 관련없는 Gaussian noise로 구성된 single-channel images이며, synthesis network의 각 레이어에 전용 노이즈 이미지(dedicated noise image)를 제공한다.

Noise image는 학습된 feature 별 스케일링 계수를 사용하여 모든 feature map으로 broadcasting된 다음 Fig. 1b에 표시된 것처럼 해당 convolution의 output에 추가된다.

Noise inputs 추가의 영향은 Section 3.2 와 3.3에 설명되어 있다.

### 2.1. Quality of generated images

  Generator의 특성을 연구하기 전에, 우리는 재설계가 이미지 퀄리티를 저하시키지 않는다는 것을 실험적으로 증명하고, 실제로 이미지 퀄리티를 상당히 개선한다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-06 at 7.19.19 PM.png" alt="Screen Shot 2021-08-06 at 7.19.19 PM" style="zoom:50%;" />

Table 1은  CELEBA-HQ의 다양한 generator architecture와 새로운 FFHQ datasets을 위한 Fréchet inception distances (FID)를 보여준다.

다른 데이터셋에 대한 결과는 Appendix E에서 보여준다. 우리의 baseline coniguration `(A)`는 Karras et al. 의 Progressive GAN 설정이며, 여기서 달리 명시된 경우를 제외한 네트워크와 모든 hyperparameters를 상속한다.

우리는 먼저 향상된 baseline `(B) `를 bilinear up/downsampling operations [64], longer training, and tuned hyperparameters를 사용하여 바꾼다. 

자세한 트레이닝 설정과 하이퍼파라미터 설명은 Appendix C에 포함된다. 

그런 다음 mapping network와 AdaIN을 추가하여 이 새로운 baseline을 더욱 개선한다 (C). 그리고 네트워크가 첫번째 convolution layer에 latent code를 입력받아도 더이상의 이득을 얻지 못한다는 놀라운 사실을 관찰했다.

우리는 따라서 전통적인 input layer를 제거하고 학습된 4 × 4 × 512 constant tensor `(D)`로부터 image sythesis를 시작함으로써 아키텍처를 간단화했다.

우리는 synthesis network가 AdaIN를 제어하는 styles을 통해서만 input을 받음에도 불구하고 의미 있는 결과를 낼 수 있다는 것을 발겼했다.



  마지막으로, 결과를 더욱 개선하는 noise input `(E)` 와 주변 styles를 decorrelates하고 생성된 imagery를 보다 세밀하게 제어할 수 있는 새로운 *mixing regualarization* `(F)` 를 소개한다(Section 3.1).

  우리의 방법을 두개의 다른 loss function을 사용하여 평가했다: CELEBA-H의 경우  WGAN-GP에 의존하는 반면, FFHQ는 configuration `A`에 WGAN-GP를 사용하고 구성 `B-F`의 경우 R1 regularization을 통한 non- saturating loss을 사용한다. 

이러한 선택이 가장 좋은 결과를 보여주었다. 우리의 contribution은 loss function을 수정하지 않는다.

  style-based generator`(E)`는 기존 generator `(B)` 거의 20%가까이 FIDs가 상당히 향상되었고, parallel work [6, 5]에서 수행되는 large-scale ImageNet 측정을 확증했다.(?) 

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-06 at 8.13.29 PM.png" alt="Screen Shot 2021-08-06 at 8.13.29 PM" style="zoom:50%;" />

Figure 2는 우리의 generator를 사용하여 FFHQ 데이터 세트에서 생성된 새로운 uncurate set을 보여준다. 

FIDs에서 확인했듯이 평균 퀄리티가 높고 안경이나 모자 같은 액세서리도 성공적으로 합성된다.

이 그림의 경우, 소위 truncation trick - Appendix B 에서 Z대신 W에서 trick이 수행되는 방법을 자세히 설명한다.- 을 사용하여  W의 극한 영역으로부터 샘플링을 피했다. 

우리의 generator는 선택적으로 저해성도에만 잘라내기(truncation)를 적용할 수 있도록 하여 고해상도 세부사항에는 영향을 미치지 않는다.

  이 논문의 모든 FIDs는 truncation trick 없이 계산되고, 우리는 Fig. 2와 video의 예시 목적(illustrative purposes)으로만 사용한다.

모든 이미지는 1024^2 해상도로 생성된다.

### 2.2 Prior art

  대부분의 GAN architectures에서의 연구는 multiple discriminators, multiresolution discrimination, self-attention 등을 사용하여 discriminator를 향상시키는 데에 초점을 두어왔다. 

Generator 측면의 연구는 대부분 input latent space의 exact distribution이나 Gaussian mixture models, clustering, ncouraging convexity를 통한 input latent space 형상에 초점을 맞춘다.

  최근 conditional generators는 별도의 embedding network를 통해 class classifier를 generator의 많은 레이어에 공급하는 반면, latent는 여전히 input layer를 통해 제공된다.

몇몇 저자는 latent code의 일부를 multiple generator layers에 input하는 것을 고려했다.

병렬 작업에서 Chen et al.은 AdaINs를 사용하여 generator를 "self modulate"하는데, 이는 우리의 작업과 유사하지만, intermediate latent space나 노이즈 입력은 고려하지 않는다.

## 3. Properties of the style-based generator

  우리의 generator 아키텍처를 통해 style에 대한 스케일 별 수정을 통해 image synthesis를 제어할 수 있다.

우리는 mapping network와 affine trainsformations은 학습된 distribution에서 각 styles에 대한 sample을 그리는 방법으로 볼 수 있고, synthesis network는 styles의 collection을 기반으로 새로운 이미지를 생성하는 방법으로 볼 수 있다.

각 styles의 효과는 네트워크에 국한된다. 즉, styles의 특정 부분집합을 수정하면 이미지의 특정 측면에만 영향을 미칠 수 있다.

  이런 localization의 이유를 보기 위해, AdaIN operation이 각 채널을 zero mean 및 unit variance로 normalize한 다음 style에 따라 scale 및 biases를 적용하는 방법을 고려해보겠다.

Style에 따른 새로운 채널 별 통계는 후속 convolution operation에 대한 features의 relative importance를 수정하지만 normalization때문에 original statistics에 의존하지 않는다.

따라서 각 style은 다음 AdaIN operation에 의해 overidding 되기 전에 오직 하나의 convolution만 제어한다.

### 3.1 Style mixing

  Styles이 localize되도록 장려하기 위해, 우리는 *mixing regularization*을 채택하는데, 이것은 주어진 비율의 이미지를 training 중 하나의 latent codes를 사용하는 대신 두 개의 random latent code를 사용하여 생성한다.

이러한 이미지를 생성할 때, synthesis network에서 무작위로 선택된 point에서 하나의 latent code 에서 다른 latent code(*style mixing*이라 하는 operation)으로 전환하면 된다.

구체적으로, 