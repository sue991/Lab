# UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

## ABSTRACT

 우리는 deep convolutional generative adversarial networks(DCGANs)라고 불리는 특정 아키텍처 제약이 있는 CNN 클래스를 소개하고 이들이  unsupervised learning의 유력한 후보임을 입증한다. 다양한 이미지 데이터셋에 대한 교육을 통해, 우리의 심층 컨볼루션 적대적 쌍(deep convolutional adversarial pair)이 generator와 discriminator 모두에서 object 일부분에서 전체까지 representations의 계층 구조를 학습한다는 설득력 있는 증거를 보여준다. 또한 학습된 features를 새로운 task에 사용하여 일반적인(general한) 이미지 표현으로 적용 가능성을 입증한다.

## INTRODUCTION

  레이블링이 되지 않은 대규모 데이터셋에서 재사용 가능한 feature representations을 학습하는 것은 활발한 연구 영역이다. 컴퓨터 비전의 맥락에서, 라벨이 부착되지 않은 이미지와 동영상을 실질적으로 무제한으로 활용하여 좋은 중간 표현(intermediate representations)을 배울 수 있으며, 이미지 분류(image classification)와 같은 다양한 supervised learning에 사용할 수 있다. 좋은 image representations을 구축하는 한 가지 방법은 Generative Adversarial Networks(GANs)를 교육하고 나중에 제너레이터 및 discriminator 네트워크의 일부를 감독 작업을 위한 기능 추출기(feature extractors)로 재사용하는 것이라고 제안한다. GANs은 maximum likelihood techniques에 대한 매력적인 대안을 제공한다. 또한 그들의 learning process와 경험적 비용 함수(heuristic cost function)(ex: 픽셀 단위 independent mean-square error)가 없는 것이 representation learning에 매력적이라고 주장할 수 있다. GAN은 훈련하기에 불안정한 것으로 알려져 있으며, 종종 generator의 결과가 무의미한 output을 생산하는 경우가 있다. GANs이 학습하는 내용과 multi-layer GAN의 중간 표현(intermediate representations)을 이해하고 시각화하는 데 있어 발표된 연구는 매우 제한적이다.

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

이미지를 모델링 하기 위해 CNNs를 사용하여 GANs를 확장하려는 과거의 시도들은 성공적이지 않았다. 이는 이제 LAPGAN의 저자들이 보다 신뢰성 있게 모델링할 수 있는 저해상도 생성 이미지를 반복적으로 상향 조정하는 alternative approach를 개발하도록 동기 부여했다. 우리는 또한 supervised literature에서 일반적으로 사용되는 CNN 아키텍처를 사용하여 GAN을 스케일링하려는 시도에도 어려움을 겪었다. 그러나, 광범위한 모델 탐구 후에 우리는 많은 데이터셋에 대해서 안정적인 학습을 가져오는 아키텍처의 family를 확인하였으며 이는 더 높은 해상도와 더 깊은 생성 모델이 가능하도록 만들었다. 

우리의 접근법의 핵심은 CNN 아키텍처에 대한 3개의 최근 검증된 변화를 채택하고 수정하는 것이다.

  첫 번째는 maxpooling과 같은 deterministic spatial pooling functions를 strided convolutions으로 대체하는 모든 convolutional net은 네트워크가 이것 자체의 spatial downsampling을 학습하도록 허용한다는 것이다. 우리는 이 접근법을 generator에서 사용하였으며, 이는 우리의 spatial upsampling을 학습할 수 있도록 했고, discriminator에서도 사용한다.

  두 번째는 convolutional features의 마지막 부분에서 fully connected layer를 제거하는   쪽으로의 트렌드이다. 가장 강력한 예는 최첨단 image classification models에서 활용된 global average pooling이다. 우리느 global average pooling이 모델 안정성을 증가시키지만 수렴 속도(convergence speed)를 해친다는 것을 발견했다. 가장 높은 convolutional features를 generator와 discriminator의 input과 output 각각 직접적으로 연결하는 중간 단계가 잘 작동했다. 균일한 noise distribution Z를 입력으로 사용하는 GAN의 첫 번째 레이어는 matrix multiplication에 불과하므로 fully connected라고 할 수 있지만, 결과는 4차원 tensor로 reshaped되어 convolution stack의 시작부분으로 사용된다. Discriminator의 경우 마지막 convolution layer가 flatten되고 다음 단일 sigmoid output으로 공급된다. 모델 아키텍처의 한 예시가 Fig. 1에 나와있다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-29 at 6.13.14 PM.png" alt="Screen Shot 2021-07-29 at 6.13.14 PM" style="zoom:50%;" />

  세 번째는 Batch Normalization으로, 각 unit에 대한 input을 평균이 0이고 분산이 1이 되게끔 normalize하여 학습을 안정적으로 만들어준다. 이는 잘못된 initialization으로 인해 발생하는 교육 문제를 해결하는 데 도움이 되며 더 깊은 모델에서 gradient 흐름을 돕는다. 이는 deep generators가 학습을 시작하도록 하는 데 매우 중요한 것으로 판명되어 generator가 모든 샘플을 GAN에서 흔히 관찰되는 failure mode인 단일 point으로 collapsing되는 것을 방지했다. 하지만, 모든 layer에 직접적으로 batch norm을 적용하는 것은 sample oscillation과 모델 불안정성을 야기한다. 이것은 generator의 output layer와 discriminator의 input layer에 batchnorm을 적용하지 않는 것을 통해 피할 수 있다.

  ReLU activation은 Tanh function을 사용하는 output layer를 제외하고 generator에서 사용된다. 우리는 bounded activation을 사용하는 것이 모델로 하여금 더 빠르게 saturate 하도록 학습시키고 학습 분포(training distribution)의 color space를 포괄하도록 학습함을 발견했다. Discriminator 내에서 leaky rectified activation이 잘 작동함을 확인하였고, 특히 resolution modeling에서 잘 작동한다. 이것은 maxout activation을 사용하는 original GAN 논문과는 대조적이다.

### 안정적인 Deep Convolution GANs를 위한 Architecture guidelines

- Pooling layer를 strided convolution (discriminator)과 fractional-strided convolution(generator)으로 교체한다. 
- Generator와 discriminator에 batchnorm을 사용한다.
- 더 깊은 아키텍처를 위해 fully connected hidden layer를 없앤다.
- Generator에서 Tanh를 사용하는 output layer를 제외한 모든 layer에서 ReLU activation을 사용한다.
- Discriminator의 모든 레이어에서 LeakyReLU activation을 사용한다.

## DETAILS OF ADVERSARIAL TRAINING

  우리는 DCGANs을  Large-scale Scene Understanding (LSUN), Imagenet-1k 그리고 newly assembled Faces dataset 총 3개의 데이터셋에서 학습했다. 각 데이터셋의 자세한 사용은 밑에 나와있다.

  Tanh activation function의 범위 [-1, 1] 에 대한 스케일링 외에 training images에 대한 전처리(pre-processing)은 적용되지 않았다. 모든 모델은 mini-batch stochastic gradient descent (SGD)를 이용해서 학습되었으며, 128의 mini-batch를 사용했다. 모든 weights는 standard deviation 0.02를 가지는 zero-centered normal distribution으로부터 초기화(initialized) 되었다. LeakyReLU에서, leak의 slope는 모든 모델에서 0.2로 설정되었다. 이전 GAN 연구에서는 training을 가속화하기 위해 momentum을 사용하였으나, 우리는 튜닝된 hyper parameters를 가지고 Adam optimizer를 사용했다.  우리는 제안된 learning rate 0.001이 너무 높다는 것을 알았고, 대신 0.0002를 사용했다.  추가적으로, momentum term β1의 제안된 값 0.9가 training oscillation과 불안정을 야기하여 학습을 안정화시키고자 0.5로 줄였다.

### 1. LSUN

  generative image models의 샘플의 시각적 품질이 향상됨에 따라 training sample의  over-fitting 및 memorization에 대한 우려가 높아졌다. 우리의 모델은 더 많은 데이터와 더 높은 해상도의 생성으로 확장되는 방법을 보여주기 위해 300만개가 조금 넘는 교육 사례를 포함하는 LSUN bedrooms dataset에 대한 모델을 교육한다. 최근 분석 결과는 모델의 학습 속도와 일반화 성능(generalization performance) 사이에는 직접적인 연관이 있다는 것을 보여준다. Convergence 후 샘플(Fig.3)뿐만 아니라 online learning을 모방한 한 epoch의 sample(Fig. 2)을 보여 주며, 우리의 모델이 단순히 overfitting/memorizing training 예를 통해 고품질 샘플을 생산하지 않음을 입증할 수 있는 기회로 삼았다. 이미지에는 data augmentation이 적용되지 않았다.

### 1.1 DEUDUPLICATION

  generator가 input examples(Fig. 2)를 암기할 가능성을 더욱 줄이기 위해 간단한 이미지  de-duplication process를 수행한다. 우리는 training examples의 다운샘플링된 center-crops 32x32에 3072-128-3072 de-noising dropout regularized RELU autoencoder를 장착한다. 그 결과 발생하는 코드 layer activations는 효과적인 information preserving technique(Srivastava et al., 2014)인 ReLU activation thresholding을 통해 이진화되며, linear time de-duplication를 허용하는 편리한 형태의 semantic-hashing을 제공한다. hash collisions에 대한 육안 검사에서 estimated false positive rate가 100분의 1 미만인 높은 precision를 보였다. 추가적으로, 이 기법은 duplicates에 가까운 약 275,000개를 감지하고 제거하여 높은 recall을 제안했다.

### 2. FACES

우리는 사람 이름들의 임의의 웹 이미지 쿼리에서 사람의 얼굴을 포함한 이미지들을 스크랩했다. 그 사람들의 이름은 근대에 태어났다는 기준을 가지고 dbpedia에서 따온 것이다. 이 데이터 세트에는 10K명의 3M개의 이미지가 포함되어 있다. 이러한 이미지에서 OpenCV face detector를 실행하고 충분히 고해상도인 detections를 유지하여 약 350,000개의 face boxes를 제공한다. 우리는 이 faces boxes를 트레이닝에 사용한다. 이미지에는 data augmentation이 적용되지 않았다.

### 3. IMAGENET -1K

우리는 unsupervised training을 위한 natural images로써 Imagenet-1k를 사용했다. 우리는 32 x 32 min-resized center crops에 대해 훈련한다. 이미지에는 data augmentation이 적용되지 않았다.

## EMPIRICAL VALIDATION OF DCGANS CAPABILITIES

### 1. CLASSIFYING CIFAR-10 USING GANS AS A FEATURE EXTRACTOR

  Unsupervised representation learning algorithms의 퀄리티를 평가하기 위한 한 방법은 supervised datasets에 이를 feature extractor로 적용하는 것이고 이 features의 top에 linear models를 fitting 시켜 성능을 평가하는 것이다.

  CIFAR-10 데이터셋에서, feature learning algorithm으로 K-means를 활용하는 single layer feature extraction pipeline으로부터 매우 강력한 baseline performance가 검증되었다. 매우 많은 양의 feature maps(4800)을 사용할 경우 이 기술은 80.6%의 정확도를 달성한다. 기본 알고리즘을 unsupervised multi-layered로 확장하면 82.0%의 정확도를 달성했다. Supervised tasks를 위해 DCGANs에 의해서 학습된 representation의 퀄리티를 평가하기 위해 DCGAN을 Imagenet-1k에 학습하고 discriminator의 모든 layer에서 나온 convolutional feature를 사용하였으며 4x4 spatial grid를 만들기 위해서 각 layer의 representation을 max-pooling 했다. 이 feature들을 flatten 하고 이를 concatenate 해서 28672차원의 벡터를 만든 다음 이에 대해서 regularized linear L2-SVM classifier를 학습시킨다. 이는 82.8% accuracy를 달성했으며, 모든 K-means 기반 접근법보다 더 좋은 성능을 달성했다. 특히 discriminator는 K-means 기반의 기법과 비교했을 때 훨씬 적은 feature map을 가지지만 (가장 높은 layer에서 512개) 4x4 spatial locations의 많은 layer 때문에 더 큰 전체 feature vector size를 갖게 된다. DCGAN의 성능은 여전히 normal discriminative CNN을 감독되지 않는 방식으로 훈련시켜 소스 데이터셋에서 특별히 선택되고, 공격적으로 증강된 샘플 샘플을 구별하는 기술인 Campleanar CNNs(Dosvitski et al., 2015)보다 낮다.  Discriminator의 representations을 미세하게 조정함으로써 더 많은 개선이 이루어질 수 있지만, 우리는 이것을 향후 작업에 남겨두고 있다. 또한, DCGAN은 CIFAR-10에 대해 교육을 받은 적이 없기 때문에 이 실험은 학습된 기능의 domain robustness을 보여준다. 

### 2. CLASSIFYING SVHN DIGITS USING GANS AS A FEATURE EXTRACTOR

  SVHN(StreetView House Numbers) 데이터셋(Netzer 등, 2011)에서는 레이블이 지정된 데이터가 부족한 경우 DCGAN의 discriminator의 features를 supervised purposes으로 사용한다. CIFAR-10 실험에서와 유사한 데이터 집합 준비 규칙을 따라 non-extra set에서 10,000개의 예제로 구성된 validation set를 분리하여 모든 hyperparameter and model selection에 사용한다. 1000개의 균일 분포 training examples를 무작위로 선택하여 CIFAR-10에 사용된 것과 동일한 기능 추출 파이프라인 위에서 regularized linear L2-SVM classifier를 교육하는 데 사용한다. 이는 test error 22.48%로 최신 기술(1000개의 레이블을 사용한 classification)을 달성하여 unlabled data를 활용하도록 설계된 CNN의 또 다른 수정 사항을 개선한다. 추가적으로, DCGAN에 사용된 CNN 아키텍처가 동일한 데이터에 동일한 아키텍처를 가진 purely supervised CNN을 교육하고 64개 hyperparameter trials에 대한 무작위 검색을 통해 이 모델을 최적화함으로써 모델의 성능에 핵심적인 기여 요소가 아님을 검증한다. Validation error는 28.87%로 상당히 높다. 

## INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS

  우리는 다양한 방법으로 훈련된 generators와 discriminators를 살펴보았다. 우리는 training set에 대해 어떠한 종류의 nearest neighbor search를 하지 않았다. Pixel 또는 feature space에서 Nearest neighbors는 작은 이미지 변환에 의해 하찮게 속는다(?). 우리는 또한 모델을 정량적으로 평가하기 위해 log-likelihood metrics를 사용하지 않는데, 이는 안 좋은 metric이기 때문이다.

### 1. WALKING IN THE LATENT SPACE

 우리가 한 첫 번째 실험은 latent space의 landscape를 이해하는 것이다. 학습된 manifold에서 걷는 것은 일반적으로 우리에게 memorization의 signs에 대해서 말해줄 수 있고 (만약 sharp transitions이 있는 경우) 공간이 계층적으로 collapsed 된 방법에 대해서도 말해줄 수 있다. 만약 latent space에서 걷는 것이 이미지 생성에 대해서 의미론적인 변화를 야기하는 경우(객체가 추가되거나 제거되는 것을 의미) 우리는 모델이 관련되어 있는 표현을 학습했고 흥미로운 표현을 학습했다고 추론할 수 있다. 결과는 Fig. 4에서 볼 수 있다.

### 2. VISUALIZING THE DISCRIMINATOR FEATURES

  이전 연구는 큰 이미지 데이터셋에 CNN을 supervised training 했을 때 매우 강력한 학습된 feature를 야기한다는 사실을 보였다. 추가적으로, scene classification에 학습된 supervised CNN은 object detectors를 학습한다. 우리는 large image dataset에 학습된 unsupervised DCGAN도 역시 흥미로운 특징의 계층을 학습할 수 있음을 보인다.

Springenberg et al., 2014에 의해 제안된 guided backpropagation을 사용하면서, Fig.5에서 discriminator에 의해서 학습된 feature가 침대나 창문과 같은 bedroom의 특정한 부분에서 활성화된다는 것을 보였다. 비교를 위해서, 같은 이미지에서, 우리는 의미론적으로 관련이 있거나 흥미로운 어떤 것에 활성화되지 않은 임의로 초기화된 feature에 대한 baseline을 제공한다.

### 3. MANIPULATING THE GENERATOR REPRESENTATION

### 3.1 FORGETTING TO DRAW CERTAIN OBJECTS

  Discriminator에 의해 학습된 표현에 더하여, generator가 학습한 표현이 무엇인지에 대한 질문이 있다. 샘플의 퀄리티는 generator가 베개, 창문, 램프, 문, 그리고 이것저것 다양한 가구와 같은 주요한 장면 요소에 대한 구체적인 object representation을 학습했음을 시사한다. 이러한 표현의 형태를 탐색하기 위해 generator에서 창문을 완전히 제거하는 실험을 수행했다. 

  150개 sample에 대해, 52개의 창문 bounding box를 직접 그렸다. 두 번째로 높은 convolutino layer features에 대해, logistic regression은 bounding box 안쪽에 있는 activation은 positive로, 같은 이미지에서 랜덤한 샘플을 뽑아서 negative로 지정하는 기준을 사용해 feature activation이 window에 있는지 없는지를 예측하기 위해 fitting 되었다. 이 간단한 모델을 사용하여, 0보다 큰 weights를 가지는 모든 feature maps(총 200개)이 모든 spatial locations로부터 드랍되었다. 그러면, 임의의 새로운 샘플은 feature map을 제거하거나 제거하지 않고(feature map removal 있거나 없이) 생성된다.

  Window dropout을 가지고 만들어진 이미지와 없이 만들어진 이미지는 Fig. 6에 나와있다. 그리고 흥미롭게도, 네트워크는 대개 bedroom에 window를 그리는 것을 까먹고, 이를 다른 object로 대체했다.

### 3.2 VECTOR ARITHMETIC ON FACE SAMPLES

  단어의 학습된 학습된 representation을 평가하는 맥락에서 (Mikolov etal., 2013) 단순한 산술 연산이 representation space에서 풍부한 linear structure를 나타낸다는 것이 검증되었다. 하나의 표준이 되는 예시는  `vector("King") - vector("Man") + vector("Woman")`가 `Queen`을 의미하는 vector에 가장 가까운 vector를 야기한다는 사실을 나타낸다. 우리는 유사한 구조가 우리의 generator의 Z representation에서 나타나는지 아닌지를 조사했다. 우리는 시각적 개념에 대한 샘플 세트의 Z 벡터에 대해 유사한 산술을 수행했다. 개념 당 오직 하나의 샘플로 실험을 진행하는 것은 불안정(unstable)했으나, 3개의 예시에 대한 Z vector를 평균 내는 것은 의미론적으로 산수를 따르는 안정적이고 일관된 생성을 보여주었다. Fig. 7에 나타난 object manipulation에 더해서, 우리는 face pose 또한 Z space에서 선형적으로 모델링 될 수 있음을 검증하였다. (Fig. 8)

  이러한 검증은 우리의 모델에 의해 학습된 Z representations을 사용하여 흥미로운 application이 개발될 수 있음을 제안한다. Conditional generative models이 scale, rotation, position과 같은 object attributes를 설득력있게 모델링하는 방법을 학습할 수 있음은 이전에 증명되었다. 이는 우리가 알기로 완전히 unsupervised models에서 이러한 현상이 발생한다는 것을 증명한 첫 번째 사례이다. 위에서 언급된 벡터 산수를 더 탐구하고 개발하는 것은 복잡한 이미지 분포의 conditional generative modeling에 필요한 데이터의 양을 극적으로 줄일 수 있다.

## CONCLUSION AND FUTURE WORK

  우리는 generative adversarial networks를 학습시키는 더욱 안정적인 아키텍처 set을 제안하였으며 adversarial networks가 generative modeling과 supervised learning을 위한 good image representation을 학습한다는 것에 대한 증거를 제시했다. 여전히 모델 불안정성의 몇 가지 형태가 남아 있으며 우리는 모델이 더 오래 학습할수록 그들이 때때로 filter의 subset을 하나의 oscillating mode로 붕괴된다는 것을 알고 있다. 

  이후 연구는 이런 불안정성의 형태를 다룰 필요가 있다. 우리는 이 framework를 video (frame prediction을 위한)나 audio (speech synthesis를 위한 pre-trained feature)와 같은 다른 분야로 확장하는 것이 매우 흥미로울 것이라고 생각한다. 학습된 latent space에 대한 특성을 더 조사해보는 것 또한 흥미로울 것이다.

