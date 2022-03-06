# Analyzing and Improving the Image Quality of StyleGAN

## Abstract

  SteyleGAN은 데이터 기반 unconditional generative image modeling에서 최첨단 결과를 제공한다.

stylegan2에서는 지난 버전에서 몇 가지 특징적인 결함들을 분석했고, 이에 대한 방법으로 **모델 구조와, 훈련 방법**에 있어 변화를 제안한다.

특히, 우리는 latent codes에서 images로 mapping하는 데 더 좋은 컨디션을 만들기 위해 generator normalization을 재디자인하고, progressive growing을 재고하고, generator를 정규화했다. 

게다가 이미지 퀄리티를 향상시키기 위해, *path length regularizer*는 제너레이터가 invert되기 매우 쉬어지도록 추가적인 이점을 제공한다.

따라서 생성된 이미지를 특정 네트워크에 안정적으로 속성을 지정할 수 있다.

게다가 출력물의 해상도를 얼마나 잘 활용하는지 시각화하여 수용량의 문제를 발견했고, 이에 따라 추가적인 품질 향상을 위해 수용력이 더 큰 모델을 이용해 학습하게 됐다.

전반적으로, 우리의 개선된 모델은 기존 배포 품질 메트릭스와 인식된 이미지 품질 면에서 무조건적인 이미지 모델링에서 최첨단 기술을 재정의한다.

## Introduction

  GAN에 의해 생성된 이미지의 해상도와 퀄리티는 빠르게 향상되고 있다. 높은 해상도 이미지 synthesis를 위한 현재 최신 방법은 StyleGAN인데, 이것은 안정적으로 여러 데이터셋에서의 작동을 보여주었다. 

우리의 연구는 이것의 인공적인 결함을 고치고 결과 퀄리티를 더욱 향상시키는데 초점을 두었다.

  StyleGAN의 특색있는 특징은 색다른 generator architecture이다. 

Network의 시작 부분에 input latent code z ∈ Z 를 넣는 대신, *mapping network f* 가 먼저 이것을 intermediate latent code w ∈ W로 transform한다. Affine transforms은 그다음 *synthesis network* g 의 레이어를 컨트롤하는 *styles* 를 만드는데, AdaIN을 통해 만든다.

추가적으로, Stochastic variation이 추가적인 random noise를 synthesis network에 매핑하여 제공하면서 촉진되었다.

이 디자인이 intermediate latent space W가 input latent space Z보다 훨씬 덜 entangle하다는 것을 증명해왔다.

이 논문에서, 우리는 합성 네트워크의 관점에서 관련된 latent space인 W의 분석에 오로지 초점을 맞출 것이다.

  많은 관찰자들은 StyleGAN에 의해 생성된 이미지에 인공적인 결함이 있었다는 것을 알아차렸다.

우리는 이러한 아티팩트의 두 가지 원인을 파악하고 이를 제거하는 아키텍처 및 교육 방법의 변화를 설명한다

먼저, 일반적인 blob 형태의 artifacts의 근원을 조사하여 제너레이터가 아키텍처의 설계결함을 피하기 위해 아티팩크를 생성한다는 것을 알아낸다.

Section 2에서, 우리는 generator에 사용된 normalization을 재디자인해 artifacts를 제거한다. 

두번째로, 고해상도 GAN 훈련 안정화에 크게 성공한 점진적 성장(progressive growing)과 관련된 아티팩트를 분석한다.

우리는 대안적인 같은 목표를 달성하는 대안적인 디자인을 제시하는데, 그것은 training이 저해상도 이미지에 초점을 맞추며 시작하고 점진적으로 더 높은 해상도로 초점을 바꾸는 것이며, 트레이닝동안 네트워크의 topology를 바꾸지 않는다.

또한 이 새로운 설계를 통해 생성된 이미지의 효과적인 해상도를 추론할 수 있으며, 이는 용량 증가를 유발한다(Section 4).

  Generative methods를 사용하여 생성된 이미지의 퀄리티의 정량적 분석은 여전히 어려운 주제이다.

Fréchet inception distance (FID) 방법은 InceptionV3 classifier의 고차원 feature space에서 두 분포의 밀도 차이를 측정한다. 

Precision and Recall (P&R)은 각각 training data와 유사한 생성된 이미지의 percentage와 생성될 수 있는 training data의 퍼센트를 명시적으로 정량화하여 추가적인 visibility를 제공한다.

이러한 metrics를 향상됨을 측정하는데 사용한다.

FID와 P&R 모두 최근 shape보다는 textures에 초점을 맞춘 것으로 나타난 classifier networks를 기반으로 한다. 따라서 metrics가 이미지 품질의 모든 측면들 정확하게 포착하지는 못한다.

우리는 원래 latent space interpolations을 추정하기 위한 방법으로 소개된 Perceptual path length (PPL) metric가 shape의 일관성 및 안정성과 상관관계가 있음을 관찰했다.

이것에 근거하여, 우리는 smooth mapping을 선호하고 명확한 품질 향상을 달성하기 위해 통합 네트워크를 regularize 한다.

또한 계산 비용에 대응하기 위해 모든 정규화를 덜 자주 실행할 것을 제안하며, 이러한 작업이 실효성이 보장되지 않는지 관찰할 것을 제안한다.

  마지막으로, latent space W에 이미지를 투영하는 것이 원본 StyleGAN보다 새로운 path length regularized StyleGAN2 generator를 통해 훨씬 더 잘 작동한다는 것을 알았다.

이렇게 하면 생성된 이미지를 소스에 더 쉽게 연결할 수 있다.

model : `https://github.com/NVlabs/stylegan2`

## Removing normalization artifacts

  StyleGAN에 의해 생성된 대부분의 이미지가 water droplets를 닮은 특징적인 blob-shaped artifacts를 나타내는 것을 관찰했다. Figure 1에서 보는것 처럼, 심지어 droplet이 마지막 이미지에서 명백히 보이지 않을지라도, 제너레이터의 intermediate feature maps에서 나타난다.

그 anomaly는 64*64 해상도 즈음에서 나타나기 시작하며, 모든 feature map에 표현되고, 점진적으로 강하고 높은 해상도가 된다.

이러한 일관된 아티팩트의 존재는 discriminator가 탐지할 수 있어야 하므로 수수께끼이다.

  우리는 feature map의 평균과 분산을 각각 normalize해서 피쳐의 크기에서 발견된 정보가 서로에 대해 파괴될 수 있는 AdaIN operation에 문제가 있다고 포인트를 잡았다.

우리는 droplet artifact가 제너레이터가 의도적으로 인스턴스 정규화 이전의 signal sneaking 정보를 몰래 빼돌린 결과라고 가정한다. 즉, 통계를 지배하는 강력하고 localized spike를 생성하면 generator는 다른 곳에서 원하는 대로 signal을 효과적으로 스케일링 할 수 있다.

우리의 가설은 normalization step이 제너레이터에서 지워졌을때, droplet artifact가 완전히 사라지는 것을 발견하면서 주장되었다.

### 1. Generator architecture revisited

  우리는 redesigned normalization을 더 용이하게 하기 위해 먼저 StyleGAN generator의 몇몇 디테일을 개정한다.

이러한 변화는 품질 지표 측면에서 중립적이거나 작은 긍정적인 영향을 미친다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-17 at 1.56.13 PM.png" alt="Screen Shot 2021-08-17 at 1.56.13 PM" style="zoom:50%;" />  

Figure 2a는 원본 StyleGAN의 통합 네트워크 *g* 를 보여주고, Figure 2b에서 우리는 weights와 biases를 보여주고 AdaIN operation을 두 파트(normalization & modulation)로 나누면서 전체 자세한 diagram을 확장한다.

이것은 각 박스가 하나의 스타일이 active된 네트워크의 파트("style block")를 암시하기 위해 우리가 개념적인 회색 박스를 다시 그릴 수 있게 한다.

흥미롭게도, 원래 StyleGAN은 bias와 noise를 style block 이내에 적용하여 상대적인 영향을 현재 style의 magnitude에 반비례하도록 한다.

우리는 이러한 operation을 nomalized data에 대해 작동하는 style block 밖으로 옮김으로써 더 예측가능한 결과가 얻어지는것을 관찰했다. 게다가 이 변화 후에 표준편차만으로 normalization과 modulation이 작동하기에 충분하다는 것을 알았다.(i.e. 평균은 필요하지 않다.)

Constant input에 bias, noise 그리고 normalization을 적용하는 것도 관찰할 수 있는 결함 없이 안전하게 제거할 수 있다.

이러한 변형은 Figure 2c에서 보여주는데, 재설계된 정규화의 출발점이 된다.

### 2. Instance normalization revisited

  StyleGAN의 주요 강점 중 하나는 *style mixing*을 통해 생성된 이미지를 컨트롤하는 것인데, 이것은 inference time에 다른 latent w를 다른 레이어에 넣는 방법이다.

실제로 스타일 변조는 특정 피쳐 맵을 크기 또는 그 이상으로 증폭할 수 있다.

스타일 혼합이 작동하려면 샘플 단위로 이 증폭을 명시적으로 대응해야 한다. 그렇지 않으면 후속 layer가 데이터에 대해 의미 있는 방식으로 작동할 수 없다.

  만약 우리가 기꺼이 scale-specific controls을 희생한다면, 간단하게 normalization을 제거할 수 있을 것이고, 따라서 아티팩트를 제거하고 FID를 약간 개선할 수 있다.

이제 완전한 제어 능력을 유지하면서 아티팩트를 제거하는 더 나은 대안을 제안하겠다.

들어오는 피쳐 맵의 예상 통계를 기반으로 정규화를 수행하되 명시적인 강제 적용은 하지 않는 것이 기본 아이디어이다.

   Figure 2c 에 있는 style block은 modulation, convolution, normalization으로 구성된다.

먼저 convolution에 따른 modulation의 효과를 고려해보자.

Modulation은 convolution weights를 스케일링함으로써 대안적으로 수행될 수 있는 style에 기반한 각 input feature map을 스케일링한다. 
$$
w'_{ijk} = s_i·w_{ijk} \\

w, w' : \mbox{original & modulated weights} \\
s_i : \mbox{i번째 input feature map에 상응하는 scale} \\
j,k : \mbox{convolution의 output feature maps , spatial footprint}
$$


이제, instance normalization의 목적은 본질적으로 convolution’s output feature maps의 statistics로부터 s의 효과를 제거하는 것이다.

우리는 이 목표가 더 직접적으로 달성될 수 있다고 관찰한다.

input activation이 unit standard deviation인 i.i.d. random variables라고 가정하자.

Modulation과 convolution 후에, output actication은 다음과 같은 standard deviation을 갖는다:
$$
\sigma_j = \sqrt{\sum_{i,k} {w'_{ijk}}^2}
$$
output : 해당 weights의 L2 norm으로 스케일링된다.

이후의 normalization은 output을 다시 단위 표준 편차로 복원하는 것을 목표로 한다.

Equation 2에 근거하여, 

