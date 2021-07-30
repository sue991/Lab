# UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

## ABSTRACT

 우리는 deep convolutional generative adversarial networks(DCGANs)라고 불리는 특정 아키텍처 제약이 있는 CNN 클래스를 소개하고 이들이  unsupervised learning의 유력한 후보임을 입증한다. 다양한 이미지 데이터셋에 대한 교육을 통해, 우리의 심층 컨볼루션 적대적 쌍(deep convolutional adversarial pair)이 generator와 discriminator 모두에서 object 일부분에서 전체까지 representations의 계층 구조를 학습한다는 설득력 있는 증거를 보여준다. 또한 학습된 features를 새로운 task에 사용하여 일반적인(general한) 이미지 표현으로 적용 가능성을 입증한다.

## INTRODUCTION

  레이블이 지정되지 않은 대규모 데이터셋에서 재사용 가능한 feature representations을 학습하는 것은 활발한 연구 영역이다. 컴퓨터 비전의 맥락에서, 라벨이 부착되지 않은 이미지와 동영상을 실질적으로 무제한으로 활용하여 좋은 중간 표현(intermediate representations)을 배울 수 있으며, 이미지 분류(image classification)와 같은 다양한 supervised learning에 사용할 수 있다. 좋은 image representations을 구축하는 한 가지 방법은 Generative Adversarial Networks(GANs)를 교육하고 나중에 제너레이터 및 discriminator 네트워크의 일부를 감독 작업을 위한 기능 추출기(feature extractors)로 재사용하는 것이라고 제안한다. GANs은 maximum likelihood techniques에 대한 매력적인 대안을 제공한다. 또한 그들의 learning process와 경험적 비용 함수(heuristic cost function)(ex: 픽셀 단위 independent mean-square error)가 없는 것이 representation learning에 매력적이라고 주장할 수 있다. GAN은 훈련하기에 불안정한 것으로 알려져 있으며, 종종 generator의 결과가 무의미한 output을 생산하는 경우가 있다. GANs이 학습하는 내용과 multi-layer GAN의 중간 표현(intermediate representations)을 이해하고 시각화하는 데 있어 발표된 연구는 매우 제한적이다.

이 논문의 contribution은 다음과 같다.

- 우리는 Convolution GANs의 아키텍처 위상(architectural topology)에 대한 일련의 제약을 제안하고 평가하여 대부분의 환경에서 안정적으로 훈련할 수 있도록 한다. 우리는 이러한 종류의 아키텍처를 Deep Convolution GAN(DCGAN)이라고 부른다.
- 우리는 image classification tasks에 trained된 discriminators를 사용하여 다른  unsupervised algorithms과 경쟁적인 성능을 보여준다. 
- 우리는 GANs에 의해 학습된 필터를 시각화하여 특정 필터들이 특정 객체(obj)를 그리는 법을 배웠음을 경험적으로 보여준다.
- 생성된 샘플의 많은 의미 품질을 쉽게 조작할 수 있도록 generators가 흥미로운 벡터 산술 특성(vector arithmetic properties)을 가지고 있음을 보여준다.

## RELATED WORK

### 1. REPRESENTATION LEARNING FROM UNLABELED DATA

  Unsupervised representation learning은 이미지뿐만 아니라 일반적인 컴퓨터 비전 연구에서도 상당히 잘 연구된 문제이다. Unsupervised representation learning에 대한 일반적인 방법은 데이터에 대한 군집화(clustering)(예: K-means 사용)를 수행하고 clusters를 활용하여 분류 점수를 향상시키는 것이다. 이미지의 맥락에서 이미지 패치의 계층적 클러스터링(hierarchical clustering)을 통해 강력한 image representations을 학습할 수 있다(Coates & Ng, 2012). 또다른 일반적인 방법은 auto-encoder(convolutionally, stacked, 코드의 구성 요소와 위치를 구분, ladder structures)가 이미지를 compact code로 인코딩하고, 가능한 한 정확하게 이미지를 재구성하기 위해 코드를 디코딩하도록 트레이닝하는 것이다. 이 방복은 또한 image pixels로부터 좋은 feature representations을 보여줘왔다. Deep belief networks(Lee et al., 2009)는 또한 계층적 표현(hierarchical representations)을 학습하는 데 효과가 있는 것으로 나타났다.

### 2. GENERATING NATURAL IMAGES

  Generative image models은 잘 연구되었으며 parametric 모델과 non-parametric 모델의 두 가지 범주로 나뉜다.

Non-parametric models는 종종 기존 이미지의 데이터베이스에서 matching을 수행하며, 종종 이미지의 patches에 매칭하며 텍스처 합성(texture synthesis), super-resolution 및 in-painting에 사용되어 왔다.

이미지를 생성하기 위한 parametric models이 광범위하게 탐색되었다(예: MNIST digits 또는 텍스처 합성). 그러나, 현실 세계의 natural images를 생성하는 것은 최근까지 큰 성공을 거두지 못했다. 이미지를 생성하기 위한 다양한 표본 추출 접근 방식(variational sampling approach)(Kingma & Welling, 2013)은 어느 정도 성공적이었지만 샘플이 흐릿한 경우가 많았다. 또다른 접근 방식은 반복적인 전방 확산 프로세스(iterative forward diffusion process)를 사용하여 이미지를 생성한다. Generative Adversarial Networks는 noisy가 심하고 이해할 수 없는 이미지를 생성했다. 이 접근법에 대한 laplacian pyramid extension(Denton et al., 2015)은 고품질의 이미지를 보여주었지만, 여러 모델의 체인에 유입된 소음으로 인해 물체가 흔들리는 모습을 보여 어려움을 겪었다. Recurrent network approach(Gregor et al., 2015)와 deconvolution network approach (Dosvitskiy et al., 2014)도 최근 natural images를 생성하는 데 어느 정도 성공을 거두고 있다. 그러나 그들은 supervised tasks에 generators를 활용하지 않았다.

### 3. VISUALIZING THE INTERNALS OF CNNS

  신경망(neural networks)을 사용하는 것에 대한 지속적인 비판 중 하나는, 네트워크가 단순한 인간이 소비하는 알고리즘의 형태로 무엇을 하는지에 대한 이해가 거의 없는 블랙박스 방식이라는 것이다. CNN의 맥락에서, Zeiler et. al.(Zeiler & Fergus, 2014)은 deconvolutions과 filtering the maximal activations을 사용하여 네트워크에서 각 convolution filter의 대략적인 목적을 확인할 수 있었다. 마찬가지로, inputs에 gradient descent를 사용하면 filters의 특정 subsets을 활성화하는 이상적인 이미지를 확인할 수 있다.

## APPROACH AND MODEL ARCHITECTURE

이미지를 모델링 하기 위해 CNNs를 사용하여 GANs를 확장하려는 과거의 시도들은 성공적이지 않았다. 이는 이제 LAPGAN의 저자들이 보다 신뢰성 있게 모델링할 수 있는 저해상도 생성 이미지를 반복적으로 상향 조정하는 alternative approach를 개발하도록 동기 부여했다. 우리는 또한 supervised literature에서 일반적으로 사용되는 CNN 아키텍처를 사용하여 GAN을 스케일링하려는 시도에도 어려움을 겪었다. 그러나, 광범위한 모델 탐구 후에   우리는 많은 데이터셋에 대해서 안정적인 학습을 가져오는 아키텍처의 family를 확인하였으며 이는 더 높은 해상도와 더 깊은 생성 모델이 가능하도록 만들었다. 

우리의 접근법의 핵심은 CNN 아키텍처에 대한 3개의 최근 검증된 변화를 채택하고 수정하는 것이다.

  첫 번째는 maxpooling과 같은 deterministic spatial pooling functions를 strided convolutions으로 대체하는 모든 convolutional net은 네트워크가 이것 자체의 spatial downsampling을 학습하도록 허용한다는 것이다. 우리는 이 접근법을 generator에서 사용하였으며, 이는 우리의 spatial upsampling을 학습할 수 있도록 했고, discriminator에서도 사용한다.

  두 번째는 convolutional features의 마지막 부분에서 fully connected layer를 제거하는   쪽으로의 트렌드이다. 가장 강력한 예는 최첨단 image classification models에서 활용된 global average pooling이다. 우리느 global average pooling이 모델 안정성을 증가시키지만 수렴 속도(convergence speed)를 해친다는 것을 발견했다. 가장 높은 convolutional features를 generator와 discriminator의 input과 output 각각 직접적으로 연결하는 중간 단계가 잘 작동했다. 균일한 noise distribution Z를 입력으로 사용하는 GAN의 첫 번째 레이어는 matrix multiplication에 불과하므로 fully connected라고 할 수 있지만, 결과는 4차원 tensor로 reshaped되어 convolution stack의 시작부분으로 사용된다. Discriminator의 경우 마지막 convolution layer가 flatten되고 다음 단일 sigmoid output으로 공급된다. 모델 아키텍처의 한 예시가 Fig. 1에 나와있다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-29 at 6.13.14 PM.png" alt="Screen Shot 2021-07-29 at 6.13.14 PM" style="zoom:50%;" />

  세 번째는 Batch Normalization으로, 각 unit에 대한 input을 평균이 0이고 분산이 1이 되게끔 normalize하여 학습을 안정적으로 만들어준다. 이는 잘못된 initialization으로 인해 발생하는 교육 문제를 해결하는 데 도움이 되며 더 깊은 모델에서 gradient 흐름을 돕는다. 이는 deep generators가 학습을 시작하도록 하는 데 매우 중요한 것으로 판명되어 generator가 모든 샘플을 GAN에서 흔히 관찰되는 failure mode인 단일 point으로 collapsing되는 것을 방지했다. 하지만, 모든 layer에 직접적으로 batch norm을 적용하는 것은 sample oscillation과 모델 불안정성을 야기한다. 이것은 generator의 output layer와 discriminator의 input layer에 batchnorm을 적용하지 않는 것을 통해 피할 수 있다.

  ReLU activation은 Tanh function을 사용하는 output layer를 제외하고 generator에서 사용된다. 우리는 bounded activation을 사용하는 것이 모델로 하여금 더 빠르게 saturate 하도록 학습시키고 학습 분포(training distribution)의 color space를 포괄하도록 학습함을 발견했다. Discriminator 내에서 leaky rectified activation이 잘 작동함을 확인하였고, 특히 resolution modeling에서 잘 작동한다. 이것은 maxout activation을 사용하는 original GAN 논문과는 대조적이다.

### 안정적인 Deep Convolution GANs를 위한 Architecture guidelines



