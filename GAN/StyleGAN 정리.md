# StyleGAN

## 관련 연구 

### WGAN-GP

WGAN-GP에서 gratient penalty를 이용하여 WGAN의 성능을 개선했다.

![Screen Shot 2021-08-13 at 2.29.10 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 2.29.10 PM.png)

2017년 이후 WGAN-GP 논문이 나온 이후 많은 GAN 아키텍처가 WGAN-GP loss를 baseline으로써 사용하게 됐다.

StyleGAN 아키텍처에서 WGAN-GP loss가 사용된다.

### Progressive Growing of GANs (PGGAN = ProGAN)

메인 아이디어 : 학습 과정에서 레이어를 점진적으로 추가해나가는 방식으로 전체 학습의 과정을 진행한다.

즉, 한번에 전체 네트워크 구조를 구상한 다음 학습하는 것이 아닌 학습을 진행하며 먼저 앞쪽 레이어의 학습이 어느정도 진행이 되면 레이어를 붙여 학습이 진행되는 과정에서 점진적으로 해상도를 증가시키는 방식을 사용한다.

이런 방법을 사용했을 때 한번에 고해상도 이미지를 바로 만드는 것 보다 훨씬 안정적으로 고해상도 이미지를 만들 수 있었으며 학습 속도도 매우 빨라졌다.

그러나, 한계점은 이미지의 특징 제어가 어렵다. 그럴싸한 이미지는 만들 수 있지만, 다양한 특징들이 적절히 분리되어있지 않아 안경을 씌우거나 방향을 바꾸거나 하는 등의 컨트롤이 되지 않았다.

StyleGAN은 이런 PGGAN의 한계점을 해결하고자 했다.



## StyleGAN

### Mapping Network

특정 space에서의 vector를 다른 space의 vector로 매핑을 시켜주는 네트워크.

학습을 진행할 때 특정 차원의 Z 도메인에서 하나의 latent vector를 샘플링한 다음  네트워크에 넣어서 결과를 얻을 수 있었다.

StyleGAN에서는 이러한 방식때문에 특징이 잘 분리되지 않는다는 점을 지적하면서, latent vector Z를 W 도메인의 vector로 매핑 한 뒤에 W벡터를 사용하는 방식을 제안했다.

즉, 가우시안 분포에서 샘플링한 z벡터를 직접 사용하지 않고, 매핑 네트워크를 거쳐 먼저 W 벡터로 바꿔주고, 그 W 벡터를 네트워크에 넣어 각 특징들이 선형적으로 분리된 효과를 얻을 수 있었다.

<img src="/Users/sua/Library/Application Support/typora-user-images/image-20210813144309065.png" alt="image-20210813144309065" style="zoom:50%;" />

(a) : 실제 학습 데이터가 가지는 분포. 

이때 왼쪽 위가 비어있다. 세로축을 남성과 여성, 가로축을 머리 길이라고 볼 때, 왼쪽 위가 비어있다는 것은 머리가 긴 남성 데이터가 현재 데이터셋에 포함되어 있지 않음을 의미한다. 존재하지 않은 데이터이기 때문에 네트워크는 상대적으로 이러한 정보를 표현하기 어렵다.

그래서 가우시안 distribution을 따르는 하나의 latent vector를 샘플링해서 generator에 넣는다고 했을 때, 기본적으로 항상 가우시안 distribution에서 샘플링을 진행하기 때문에 구의 형태의 분포에서 샘플링을 진행하는것과 같은데, 

(b)에서 왼쪽아래 빨간색은 머리가 긴 여성으로, 오른쪽 부분은 머리가 짧은 남성이라고 했을 때 이러한 둘 사이를 interpolation 했을 때, 갑작스럽게 특징들이 바뀌는 문제를 경험하기가 더 쉽다. 육안으로 봐도 구면 자체가 linear하지 않고 curved 되어 꼬여있는 것을 확인할 수 있다. 이런 것을 말 그대로 entangled 되어있다라고 표현한다.

본문에서는 가우시안 분포에서 하나의 latent vector 를 샘플링하는 것 자체가 이런 현상을 더욱 심하게 만들 수 있다고 지적하면서, latent vector Z를 가우시안 분포에서 뽑은 뒤 바로 사용하지 않고 별도의 매핑 네트워크를 거쳐 latent vector W로 바꾼 뒤 W  space에 존재하는 vector를 이용해 이미지를 만들도록 하면 훨씬 각각의 특징들에 대해 interpolation을 수행할 때 linear space를 따라갈 확률이 높아진다고 언급한다.

즉, W 도메인을 쓰게 되면 특정 분포를 따라야 하는 제약이 사라지기 때문에 상대적으로 더 linear한 space에서 특징들이 분리되는 형태로 학습될 수 있다는 점을 강조하고 있다.

W space에서 factor들은 더욱 linear한 성질을 가질 수 있다. 그래서 StyleGAN에서는 총 8개의 layer로 구성된 하나의 매핑 네트워크를 학습해서 이러한 매핑이 수월하게 수행될 수 있도록 만들었다.

### Adaptive instance Normalizetion (AdaIN)

논문에서 하나의 이미지가 여러개의 semantic한 style 정보로 구성되는 형태로 이미지를 생성할 수 있도록 아키텍처를 구성했다.

각 스타일 정보를 어떻게 입힐 수 있냐 하면, AdaIN이라는 레이어를 활용했다.

AdaIN의 핵심 아이디어는 다른 원하는 데이터,혹은 스타일 이미지로부터 하나의 스타일 정보를 가져와서 현재 이미지에 대한 스타일 정보를 바꾸도록 만드는 것

이때 다른 정보로부터 스타일을 가져오기 때문에 학습시킬 별도의 파라미터가 필요하지 않다.

하나의 이미지를 생성할 때 여러개의 스타일 정보가 레이어를 거칠 때마다 입혀질 수 있도록 하는 방식으로 다채로운 이미지를 생성할 수 있다.

![Screen Shot 2021-08-13 at 2.58.54 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 2.58.54 PM.png)

왼쪽 : mapping network를 통해 latent vector W를 생성하고, 이게 매번 style 형태로써 실제 생성 네트워크에 들어가게 된다.

이때 latent vector W는 별도의 affine transformation을 거쳐 실제 adaIN에 들어가는 style 정보가 될 수 있도록 만든다.

생성 네트워크는 저해상도 이미지에서 출발하여 점차 너비와 높이를 증가시키는 방향으로 결과적으로 고해상도 이미지를 만드는 방식으로 구성된다.

총 9개의 block이 존재하고, 각 블럭들은 두개의 Conv layer, 두개의 AdaIN layer를 가지고 있고, 이때 AdaIN의 동작 방식은 오른쪽 그림과 같다.

실제 각 블록을 확인해보면 너비와 높이를 증가시키기 위해 Unsampling이 진행되고, 이어 Conv 수행. -> AdaIN -> Conv -> AdaIN .

이걸 반복하여 점차 해상도를 높여가며 점점 그럴싸한 이미지를 만들 수 있다.

이때, AdaIN은 컨볼루션 연산 수행 결과를 처리하기 위한 목적으로 사용되며, 각 AdaIN layer에는 Style 정보다 들어가는 것을 확인할 수 있다.

자세하게 살펴보면, Conv의 결과를 n개의 채널로 구성된 하나의 텐서라고 할 때, 이때 각각의 feature map들, 즉 각각의 채널에 대한 feature를 xi라 하자.

이때 하나의 512차원 W이 있을 때 이건 A라고 불리는 하나의 Affine transformation을 거쳐 2*n 크기의 결과를 만든다. 이는 각 채널마다 두개씩 스타일 정보를 만들어내겠다는 의미인데, 각각의 feature map에 대하여 정규된 값에 대해 얼마만큼 스케일링하고 바이어스를 더할지 설정해주는 방식으로 동작한다. 이때 y값이 style 정보이다.

 즉, 하나의 feature 정보가 있을 때, 이러한 feature의 statistics를 바꾸기 위하여 scaling을 수행하고 bias를 더하는것이다.

각 채널에 대해 정규화를 수행한다는 것은 모든 activation 값인 x에서 평균값을 빼주고 , 표준편차로 나누어줌으로써 정규화 수행할 수 있다. 

이렇게 정규화를 수행한 뒤 별도의 스케일과 bias를 적용해줌으로써 feature의 mean과 variance 값을 바꿔줄 수 있다는 것이다.



StyleGAN에서는 input 이미지가 하나의 latent vector로부터 시작되도록 하지 않아도 다양한 스타일을 추가함으로써 충분히 좋은 결과를 낼 수 있기에 초기 입력을 상수(constant)로 대체한다. => 4 * 4 * 512 차원의 하나의 학습 가능한 텐서로부터 출발하도록 만들어서 점진적으로 스타일을 입히고 Upsampling과 컨볼루션을 거쳐 네트워크를 구성한다.

## Stochastic Variation

또한 단순히 Style 정보만 입히는 것이 아니라 Stochastic variation또한 처리할 수 있는 형태로 네트워크를 구성했다.

하나의 이미지에 포함될 수 있는 다양한 확률적인 측면을 컨트롤할 수 있도록 만드는 것.

일반적으로 하나의 사람을 여러번 찍었을 때 바람의 세기에 따라 머리카락의 배치가 달라지거나 컨디션에 따라 주근깨, 여드름 등 다양한 확률적인 feature가 개입될 수 있다. 이러한 stochastic variation을 컨트롤할 수 있도록 하기 위해 별도의 noise 인풋을 넣는다.

noise  vector 또한 별도의 affine transformation을 거쳐 각 feature map에 대해 noise값이 적용될 수 있도록 만듦.

마찬가지로 각 block마다 noise가 들어갈 수 있도록 한다. 정확히는 AdaIN layer 직전에 들어간다.



스타일 인풋은 high-level global attributes를 컨트롤하기에 적합한 형태로 모델이 학습되는 반면, 

노이즈는 stochastic한 variation을 컨트롤할 수 있는 형태로 학습된다.

그래서 스타일을 바꾸면 얼굴형, 포즈, 안경의 유무 등을 바꾸는 등을 바꾸지만 노이즈를 바꾸면 주근깨, 피부모공 등을 바꿀 수 있다.

이때 앞쪽 레이어에 해당하는 coarse noise는 큰 크기의 배경 변화를 이끌어내고, Fine noise는 더 세밀한 머리 곱슬거림이나 배경과 같은 정보를 컨트롤한다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 3.26.41 PM.png" alt="Screen Shot 2021-08-13 at 3.26.41 PM" style="zoom:50%;" />

(a) : 모든 레이어에 대해 노이즈 적용

(b) : 아예 nosie 적용 X ==> 스타일을 제외한 세밀한 stochastic한 정보를 컨트롤 할 수 없음

(c) : Fine layer(뒤쪽)에만 noise 적용 => 조금 더 세밀한 머리 곱슬거임 등의 정보가 반영

(d) : Coarse layer(앞쪽)에 noise 적용 => 더 큰 큐모의 stochastic한 정보가 다뤄짐

## 아키텍처

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 3.29.34 PM.png" alt="Screen Shot 2021-08-13 at 3.29.34 PM" style="zoom:50%;" />

(a) : 기존의 전통적인 아키텍처 => PGGAN의 형태를 가져온 것

(b) : mapping network를 추가하고 스타일과 노이즈가 반영되는 형태

 latent vector Z를 샘플링한 뒤 정규화를 거쳐 W space로 매핑되고, 이러한 W space는 18개로 복제되어 각각 affine transformation 을 거쳐 각 블록마다 두번마다 스타일이 들어감.

이미지가 생성자를 거치며 고해상도의 이미지가 생성됨

또한 이런 과정에서 stochastic variation 이 들어갈 수 있도록 noise를 추가함

W vector를 사용할 경우 sub space가 linear하게 구성될 수 있기 때문에 styleGAN의 생성자는 더욱 linear하며 disentangled되어있다.

## Latent vector Meanings of StyleGAN

또한 스타일 정보가 입력되는 레이어의 위치에 따라 영향력의 규모가 달라짐을 언급한다.

디테일하게 세가지의 스타일로 나뉘는데, Coarse styles, Middle Styles, Fine styles이다.

Coarse styles는 세밀하지 않지만 전반적인 semantic features를 바꿀 수 있는 스타일을 의미한다.

정확히는 하나의 latent vectordls W는 18 * 512차원의 행렬로 볼 수 있는데, 앞쪽 레이어에 들어가는 총 4개의 latent vector가 Coarse style로 분류된다. 그 뒤 4개는 middle style, 마지막 10개는 fine style이다.

![Screen Shot 2021-08-13 at 3.35.50 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 3.35.50 PM.png)



앞쪽 Coarse style은 얼굴의 포즈나 얼굴형, 안경의 유무 등 전반적인 큼지막한 정보를 담당하고 있고,

middle style은 눈뜨고감은 여부,  헤어스타일 등 조금 더 세밀한 스타일을 담당하고, 

Fine style은 색상, 미세한 구조 등의 스타일을 담당하고 있다.

이 그림은 이미지 A가 있을 때, 특정 semantic한 스타일을 B 이미지에서 가져와서 바꾸는 모습이다.

예를들어 첫번째는 여성의 그림에서, coarse style 정보를 B의 이미지에서 가져와서 반영한 것을 알 수 있다. 따라서 latent vector w는 여성의 이미지의 그림에 대한 정보를 담고있는데 이 중에서 위쪽 coarse style 부분 만 남성의 이미지에서 가져온 것으로 바꾼 것이다.

## Evaluation

정량적 평가로 전통적으로 GANs 성능을 평가하는 방법인 FID를 사용하여 styleGAN의 성능을 평가하였다.

StyleGAN은 PGGAN을 베이스라인으로 다양한 테크닉을 적용하면서 성능이 얼마나 좋아질 수 있는지를 평가하는데, 

![Screen Shot 2021-08-13 at 3.43.17 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 3.43.17 PM.png)

다양한 기법들이 추가되면서 성능이 더 좋아지는 것을 확인할 수 있다.

셀럽A dataset과 FFHQ 데이터셋에서 styleGAN의 다양한 테크닉이 추가될수록 성능이 향상되는 것을 볼 수 있다.

A : PGGAN

B : Upsampling, Downsampling할 때 Bilinear와 같은 interpolation 테크닉을 적용할 수 있다는 것을 보여줌

C : Mapping network와 AdaIN을 적용하여 성능 향상

D : 초기 인풋 레이어로 학습된 4 * 4 * 512 상수 텐서 값을 사용

E : 노이즈 입력

F : Mixing Regularization => Style mixing 방법

## Style Mixing (Mixing Regularization)

인접한 레이어간의 스타일 상관관계를 줄인다.

다양한 스타일들이 서로 잘 분리되도록 적용하는 것이다.

**방법**

1. 두개의 입력 벡터를 준비한다.

2. crossover를 수행하는데, 이때 crossover는 벡터간의 값을 interpolation 하는 것이 아니라, 특정 포인트를 기점으로 위쪽은 하나의 vector를 쓰고 아래쪽은 다른 벡터를 쓰는 방식이다.

   <img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 3.48.48 PM.png" alt="Screen Shot 2021-08-13 at 3.48.48 PM" style="zoom:50%;" />

이렇게해서 좋은 결과가 나올 수 있는 이유는 각각의 레이어, 특히 인접한 레이어에서 서로 상관관계가 줄어들 수 있도록 유도할 수 있는 것이다.

다시말해 스타일이 각 레이어에 대하여 지역화(localized)되기 때문에 인접한 스타일 간의 상관관계를 줄일 수 있다는 것이다.



## Disentanglement 관련 두가지 성능 측정 지표 제안

1. **path length** : 두 벡터를 interpolation 할 때 얼마나 급격하게 이미지 특징이 바뀌는지.

   만약 두 이미지에 대해 Interpolation할 때 서서히 바뀌는 것이 아니라 급격하게 다양한 semantic feature가 바뀌게되면, 훨씬 덜 그럴싸한 이미지 보간(interpolation)을 수행한다고 하는 것.

   이미지가 잘 disentangled 되어 있으면 이미지가 서서히 그럴싸한 형태로 바뀌게 되므로 이러한 Path length를 측정해서 각각의 특징들이 얼마나 잘 선형적으로 분리되어있는지 평가할 수 있다.

2. **Separability : latent space**에서 각각의 Latent vector가 나타내는 attributes가 얼마나 선형적으로 분류될 수 있는지 평가



따라서 FID 뿐만 아니라 두가지 새로운 평가 지표에서도 styleGAN이 좋은 성능을 보여주는 것을 확인

또한 W space에 매핑을 한 다음 interpolation 하는 것(스타일 정보를 바꾸는 것)이 Z space에서 latent vector를 뽑아 컨트롤하는 것보다 훨씬 이상적인 결과를 낼 수 있다는 것을 알 수 있다.

 ## Latent vector interpolation

논문에서 latent vector 사이에 interpolation을 수행할 때 두가지 보간법을 이용해 평가를 진행했다.

1. **Linear interpolation(LERP)**  : 일반적인 선형보간법. 직선을 그은 뒤 그 직선 사이에 다양한 vector를 이용
2. **Spherical Linear Interpolation (SLERP) :** 구형 선형보간. 구면에 존재하는 다양한 vector들을 샘플링하는 방식으로 interpolation 수행

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 3.58.46 PM.png" alt="Screen Shot 2021-08-13 at 3.58.46 PM" style="zoom:50%;" />



## Perceptual Path Length

두개의 latent vector를 보간할 때 얼마나 급격하게 바뀌는지 체크하는 것.

정확히는 t 지점과 t + e 지점 사이에서의 사전학습된 VGG상의 특징(features)의 거리가 얼마나 먼지 계산하는 방법으로 perceptual path length 계산

Z space에서의  perceptual Path Length를 구할  때, z1과 z2를 뽑은 다음 interpolation을 수행할건데, 특정 지점이 있고 그 지점에 바로 인접한 지점이 있다고 할 때 그 두개의 지점에 대해 각각의 결과 이미지를 만들어 내고, 그러한 결과 이미지를 사전학습된 VGG network에 넣어서 특징간의 거리가 얼마나 먼지를 구하는 방식으로 거리를 계산한다.

이러한 거리를 계산함으로써 interpolation을 수행할 때 perceptual한 정보가 급격하게 바뀌는지를 체크할 수 있다.

예를 들어 다양한 feature가 얽혀 있어 안경을 쓰지않은 여성과 쓰지않은 남성 사이에서 interpolation을 수행할 때 갑자기 안경이 생겼다가 사라지는 등의 문제가 발생하는 경우 perceptual path가 크게 나오게된다.

따라서 Perceptual Path Length는 우리가 의도한 semantic한 feature만 lienar하게 잘 바뀌는지를 측정할 수 있는 지표라 할 수 있다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 4.03.38 PM.png" alt="Screen Shot 2021-08-13 at 4.03.38 PM" style="zoom:50%;" /> 

Z vector는 가우시안 분포에서 샘플링을 진행하기 때문에 구면선형보간법을 이용한다.

W vector에서는 단순히 선형 보간을 이용하여 구한다. f는 mapping network이기 때문에 f(z1), f(z2)는 각각 w1,w2라 표현할 수 있다.



## Linear Separability

선형적으로 얼마나 잘 분류될 수 있는가를 평가하는 지표

**celebA-HQ:** 얼굴마다 성별이나 웃고있는 여부 등 40개의 binary attributes가 명시되어 있는 데이터셋

이를 이용해 40개 attributes마다 각각 분류 모델을 학습한다.

 하나의 속성마다 20만개의 이미지를 생성하여 분류 모델에 넣고, confidence가 높은 10만개만 챙긴다. 이것을 학습 데이터로 사용한다.

매 attribute마다 latent vector 상에서 linear SVM 모델을 학습한다.

이때 전통적인 GAN에서는 z space에서 동작하는 linear classifier를 학습하고, StyleGAN 에서는 w space에 대해 학습을 진행한다.

그래서 각각 linear SVM 모델을 활용하여 엔트로피를 이용해 식을 구한다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 4.20.40 PM.png" alt="Screen Shot 2021-08-13 at 4.20.40 PM" style="zoom:50%;" />

하나의 이미지 데이터가 특정 클래스로 분류되기 위해서 얼마나 해당하는 feature가 부족한지에 대한 정보가 담긴다.

만약 엔트로피값이 크다면 각각의 attribute에 대한 latent vector가 linear하게 분리가 되지 않는다는 것이다.

즉, 값이 낮을수록 이상적이다!!

실제 결과를 확인했을 때, W space에서 데이터를 처리했을 때 훨씬 선형적으로 잘 분리될 뿐만 아니라 성능이 모든 성능 지표에서 잘 나온다는 것을 보여준다.

 <img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-13 at 4.22.38 PM.png" alt="Screen Shot 2021-08-13 at 4.22.38 PM" style="zoom:50%;" />



동일한 세팅으로 LSUN Bedroom 데이터셋과 LSUN Car 데이터셋에서 실험했을 때, 고해상도의 이미지가 잘 만들어지는 것을 확인했다.

또한 Coarse, Middle, Fine 각각의 스타일에 대해서도 비슷한양상을 보였음.

Coarse : 카메라 구도 등 전반적인 semantic한 feature 담당

Fine : 세밀한 색상, 재질 등 담당

