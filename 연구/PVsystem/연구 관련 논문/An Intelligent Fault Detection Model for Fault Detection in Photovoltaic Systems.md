# An Intelligent Fault Detection Model for Fault Detection in Photovoltaic Systems

[태양광발전시스템 고장감지를 위한 지능형 고장탐지모델]

2020년 9월 발표

  PV system에서 효과적인 고장 진단은 다른 환경 상태에서 current/voltage(I/V) parms를 이해하는 것이 중요하다. 특히 겨울철에는 PV 시스템에서 특정 결함 상태의 I/V characters가 정상 상태의 I/V characters와 매우 유사하다. 따라서 정상적인 고장 감지 모델은 정상 작동하는 PV 시스템을 결함 상태로 잘못 예측할 수 있으며, 그 반대의 경우도 마찬가지다. 이 논문에서는 PV 시스템의 고장 감지 및 분류를 위한 지능형 고장 진단 모델을 제안한다. 실험 검증(experimental verification)을 위해 다양한 고장 상태 및 정상 상태 데이터 세트가 광범위한 환경 조건에서 겨울철에 수집된다. 수집된 데이터셋은 몇 가지 데이터 마이닝 기법을 사용하여 정규화하고 사전 처리한 다음 probabilistic neural network(PNN)에 공급된다. PNN 모델은 새 데이터를 가져올 때 결함을 예측하고 분류하기 위해 과거 데이터로 교육된다. 기계 학습의 다른 분류 방법과 비교할 때, 훈련 받은 모델은 예측 정확도에서 더 나은 성능을 보였다.

## Introduction

  Fault detection 및 적시 문제 해결(timely troubleshooting)은 태양광 발전(PV) 시스템을 포함한 모든 발전 시스템의 최적 성능을 위해 필수적입니다. 특히, 모든 상업용 발전소의 목표는 전력 생산 극대화, 에너지 손실 및 유지관리 비용 최소화, 시설 안전 운영이다. PV 시스템은 다양한 고장과 실패가 발생하기 때문에 그러한 고장과 실패의 조기 감지는 목표를 달성하는 데 매우 중요하다[1-3]. 미국 국가 전기 법규는 특정 고장으로부터 보호하기 위해 PV 설비에 OCPD(과전류 보호 장치) 및 GFDI(접지 고장 감지 인터럽터)를 설치해야 한다. 그러나 2009년 베이커스필드 화재 사례와 2011년 Mount Holly는 이러한 장치가 특정 시나리오에서 고장을 감지할 수 없음을 보여준다[4]. PV 시스템의 고장은 물리적, 환경적 또는 전기적 조건에서 발생할 수 있다. PV array fault detection를 위한 광범위한 기술이 존재하며, 가능한 솔루션을 제공하기 위해 광범위한 연구가 수행되었다. PV 시스템의 성능을 결정하는 데 가장 중요한 두 가지 파라미터는 전류(current)와 전압(voltage)이다. I/V curves에 유도된 변형을 고려하여 결함이 있는 각 모듈과 어레이의 전기적 시그니처(electrical signature)를 고정하는 간단한 current-voltage analysis method가 제안되었다. 또 다른 연구는 mismatch fault의 정량적 정보를 추출하기 위해 PV 시스템의 전기(electrical) 및 열(thermal) 모델이 결합된 적외선 온도 측정기의 사용을 보여준다. 유사한 연구는 PV blocks의 손상 감지를 위한 항공 적외선 온도 측정(aerial infrared thermography) 및 PV 시스템 효율성 평가를 위한 현장 적외선 온도 측정 기법(onfield infrared thermography-sensing technique)을 적용한 것을 보여준다. 마찬가지로 PV 시스템의 고장 감지에 반사측정법(reflectometry methods)도 사용되었다. 단락(short circuit: 전선이 붙어버린 현상) 및 절연결함(insulation defects)를 위한 TDR(Time Domain Reflectometry) 방법이 사용되었으며, 최근에는 PV시스템에서 지락(ground faults: 접지 사고)과 노후된 임피던스(교류회로에서 전류가 흐르기 어려운 정도) 변화를 검출하기 위한 spread spectrum TDR (SSTDR) 방법이 조사되었다. 그 외에도, 아크 결함(arc faults)을 감지하기 위한 wavelet decomposition techniques 및  line-line 결함 검출을 위한 multiresolution signal decomposition의 적용은 문헌에서도 확인할 수 있다. 최근 논문에서는 PV 시스템의 몇 가지 진보된 fault detection approaches에 대한 포괄적인 연구를 제공했다. 이 연구는 fault detection approaches을 model-based difference measurement(MBDM), real-time difference measurement(RDM),output signal analysis(OSM) 및 machine learning techniques(MLT)으로 나누었다. 또한 이러한 진보된 기술과 기존 방법을 비교하여 장단점을 제시했다. 

 오늘날 대부분의 PV 시스템은 모니터링 시스템과 함께 구축되며 대용량 기간별 데이터가 지속적으로 백업된다. 인공지능(AI) 방법은 데이터 기반이며 PV 시스템에서 빅데이터를 사용할 수 있게 되면서 이 분야에 대한 연구가 탄력을 받고 있는 것으로 보인다. 특히, Machine Learning(ML-) 기반 알고리즘과 기술이 제안되며, 여기서 모델은 결함을 예측하고 분류하기 위해 과거 데이터로 교육된다. 최근 연구에 따르면 PV 모듈의 결함 분류( fault classification)를 위한 열전도(thermography)와 ML 기법이 적용된 것으로 보고되었다. 그 연구에서는 다양한 결점 열 패널 (fault panel thermal images)의 특징을 연구하기 위해 텍스처 특성(texture feature) 분석을 채택했으며 개발된 알고리즘은 93.4% 정확도로 교육되었다. 또 다른 연구는 PV 시스템에서 결함 감지, 분류 및 localization을 위한 ML 기법의 적용을 보고한다. 그 연구에서는 100%의 예측 정확도로 알고리즘을 개발했다고 주장한다. 마찬가지로, 또다른 연구는 wavelet-based approach 및 radial basis function networks (RBFN)을 활용하여 인버터의 단락(circuit) 및 단선 고장(open circuit faults)을 감지한다. 그 연구들은 1kW single-phase stand-alone PV시스템에서 테스트 했을 때, 100% training 효율성과 97% test 효율성을 보여준다.

 ML기술을 사용하는 PV 시스템을 위한 훈련된 모델의 성능은 새 데이터를 다른 환경 조건, 특히 겨울철의 데이터로부터 가져오는 경우 크게 달라질 수 있다. 겨울철 조사(irradiation) 수준은 여름보다 훨씬 낮으며, 연구는 이러한 낮은 조사(irradiation) 수준에서 발생하는 결함이 감지되지 않은 상태로 남아있을 가능성이 더 높다고 보여주었다. 이러한 감지되지 않은 faults는 상당한 양의 전원 손실(power losses)과 패널 품질 저하를 야기하거나 심지어 패널 열화(deterioration)를 초래할 수 있다.  우리는 고장 모듈을 감지하고 모든 환경 조건에 적용되는 고장 유형을 추가로 분류하기 위한 지능형 고장 진단 모델을 제안한다. 모델은 MLP를 사용하고 Supervised Learning approach를 따른다. 다양한 환경 조건, 특히 겨울에 초점을 맞춘 다양한 결함 및 정상 상태의 과거 데이터로 강력하게 훈련된다. 데이터는 전라북도에 위치한 1.8kW의 송전망 연결(grid-connected) PV 시스템에서 수집되었다. 

 본 문서의 나머지 부분은 다음과 같이 구성되어 있다. 섹션 2는 PV 시스템 고장의 개요를 소개합니다. 섹션 3은 고장 진단 모델(fault diagnosis model)의 전체 시스템 아키텍처를 설명한다. 섹션 4는 실험 결과를 제시하고, 모델을 기존 분류 방법과 비교하며, 기타 관련 문제를 논의한다. 마지막으로 섹션 5는 article을 요약하고 마무리한다.

## Overview of PV System Faults

 PV 시스템에서 발생하는 결함의 분류는 다양한 측면에서 분류할 수 있다. 이러한 결함은 물리적, 환경적, 전기적 세 가지 유형으로 분류된다. 하지만, 결함의 분류는 위치 및 구조와 같은 다른 기초에서도 이루어질 수 있다. 물리적 고장은 내부 또는 외부일 수 있으며 일반적으로 PV 모듈의 손상, 균열 및 열화(degradation)가 포함된다. 또한 PV 시스템 장애는 물리적 현상인 노화 효과에 의해 발생한다. 환경적 결함에는 토양 및 먼지 축적, 새 낙하물(bird drops), 임시 차양(temporary shading) 등이 포함된다. 영구적 환경적 결함에는 설치 위치 선택의 잘못으로 인한 영구 차양(permanent shading)이 포함된다. PV 모듈의 핫스팟 고장(Hotspot faults)은 영구 및 임시 음영(shading)으로 인해 발생할 수 있다. 마지막으로 전기적 고장에는 PV 모듈, 어레이 또는 전체 시스템의 단선, 라인 및 접지 고장이 포함됩니다. 단선 고장(Open circuit)은 PV 회로의 단일 또는 여러 분기에서 와이어가 분리되어 발생한다.

 라인 결함(Line-line faults)은 PV 어레이에서 의도하지 않은 저임피던스(low impedance) 전류 경로에 의해 생성된다. 접지 고장(ground faults)은 라인 고장과 유사하다; 그러나 저임피던스 경로는 전류 감지 도체(current-carrying conductors)에서 ground/earth까지의 경로이다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-02 at 5.47.43 PM.png" alt="Screen Shot 2021-07-02 at 5.47.43 PM" style="zoom:50%;" />]\

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-02 at 5.47.34 PM.png" alt="Screen Shot 2021-07-02 at 5.47.34 PM" style="zoom:50%;" />



Figure 1은 classification of PV array faults를, Figure 2는 PV 시스템의 main types of electrical faults을 보여준다.

  PV 모듈은 한 개의 다이오드(diode: 전류를 한 방향으로만 흐르게 하고, 그 역방향으로 흐르지 못하게 하는 성질을 가진 반도체 소자) 또는 두 개의 다이오드 모델로 전기적으로 모델링할 수 있다. 그러나 실제 PV 시스템 모델링은 PV 모듈(치수(dimension), 재료 및 접지 연결(ground connection)), 현장(site), 및 물리적 레이아웃의 변화로 인해 PV 시스템 간에 크게 다르기 때문에 매우 복잡하다. 특히 대규모 발전 시스템에서는 시스템 모델링에 특별한 기술적 과제가 따른다. 이 연구에서는 전기적 고장(electrical faults)만 감지하도록 작업을 제한했다.

![Screen Shot 2021-07-02 at 10.13.47 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-02 at 10.13.47 PM.png)

## Proposed System Architecture

이 챕터에서는 제안된 고장 진단 시스템 아키텍처를 구성하는 몇 가지 단계를 자세히 설명한다. Figure 3은 블록 다이어그램과 제안된 아키텍처의 각 단계 흐름을 보여준다. 

### 3.1 Data Acquisition 

이건 시스템 아키텍처의 첫번재 레이어이다. 모델을 만들기 위해,  PV 어레이에 연결된 각 센서로부터 전류, 전압, 조사 수준 및 온도 데이터를 수집했다. 센서는 5V level에서 동작하지만, 이 연구에 사용된 PV 모듈의 단선 전압(open circuit voltage: V_oc)은 39V고, 단락 전류(sort circuit current: I_sc) 사양은 9A이다. 활성 아날로그 필터(Active analog filters)는 PV 패널에서 전류 및 전압 센서에 주입될 수 있는 노이즈 레벨을 제거하는 데 사용되었다. 조사 수준 데이터는 0.01 ~ 200 klux 범위와 ±2%의 오류율을 가진 상용 럭스 미터(LX1330B)를 사용하여 수집되었다. 모듈에 부착된 센서에서 온도 데이터를 수집했다. 외부 온도와 패널 온도의 차이는 섭씨 1~7도(°C)였다. 모델을 교육하기 위해 가져온 입력 데이터는 각 모듈에서 측정된 평균 온도다. 데이터 세트는 여름과 겨울에 가능한 모든 환경 조건에서 수집된 데이터로 구성된다. 수집된 데이터는 클라우드 서버와 로컬 서버에서 백업되었다.

![Screen Shot 2021-07-02 at 10.14.16 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-02 at 10.14.16 PM.png)

### 3.2 Data Preprocessing

Data preprocessing은 제안된 시스템 아키텍쳐의 두번째 레이어이다. features 추출을 위해 데이터 input을 모델에 넣기 전에 수행한 모든 작업으로 구성된다. Figure 4가  본 연구에서 사용된 MLP 모델의 기능 블록(functional block)을 보여준다. fault detection 모델을 생성하기 위해 7개의 PV 데이터 features를 input layer의 input attributes로 선택한다. 

  x1은 PV 시스템의 branch 1에서 전류(A), x2는 PV 시스템의 branch 2에서 전류(A), x3은 PV 시스템의 branch 1에 있는 전압(V), x4는 PV 시스템의 branch 2에 있는 전압(V), x5는 조사 수준(irradiation level: klux), x6은 각 모듈의 평균 온도(°C), x7은 날씨상태(sunny, snowy, cloudy, and rainy)이다.

  인풋 데이터에 따르면, x7은 범주형 성격이고, 따라서, 적절한 수치 데이터로 인코딩된다. "sunny"는 1로, 나머지("snowy","cloudy", "rainy")는 0으로 인코딩된다. 그 후 모든 입력 데이터는 다음과 같이 표준화(normalize)된다.
$$
z = \frac{x-u}{s} \\
z : \mbox{sample x의 표준 점수(standard score)} \\
u : \mbox{training samples 의 평균 }\\
s : \mbox{training samples 의 표준 편차}
$$
전체 데이터셋은 training set과 test set 8:2 비율로 쪼개진다.



### 3.3 MLP & Feature Extraction

  MLP 또는 probabilistic neural network(PNN)은 ML의 비선형 학습 알고리즘(nonlinear learning algorithm)이며 supervised와 unsupervised learning 모두에 광범위하게 적용된다. 그러나 application의 대부분은 감독 학습의 classification 문제에서 발견된다.
$$
Φ_{ij}(y)  = \frac{1}{(2π)^{\frac{1}{2}}ω^d}\frac{1}{d}\sum^d_1e^{-\frac{(y-y_{ij})(y-y_{ij})^T}{ω^2}} \\

Φ_{ij}(y) \mbox{: input vector y의 probability density function, } \\
d : \mbox{training samples의 전체 카테고리 수 } \\
y_{ij} : i^{th}\mbox{sample type의 }j^{th}\mbox{ training center } \\
ω : \mbox{smoothing factor}
$$

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 12.35.39 AM.png" alt="Screen Shot 2021-07-03 at 12.35.39 AM" style="zoom:50%;" />

Figure 5는 feedforward multilayer perceptron을 보여준다.

  n0 뉴런을 가진 input layer를 사용했다고 가정하면, input layer X는 다음과 같이 지정할 수 있다.
$$
X = (x_0, x_1, ... , x_n)
$$
  feature extraction을 위해, hidden layer는 두 개의 layer로 고안되었다 : h1,h2

각각 input dimensions(x1~x7)은 h1에 공급되고, h1의 output은 h2에 공급된다.

Hidden layer에 있는 뉴런의 output h^j_i는 다음과 같이 계산된다.
$$
h^j_i = f\Bigg(\sum^{n_{i-1}}_{k=1}W^{i-1}_{k,j}h^k_{i-1}\Bigg) , i=2,...,N ; j=1,...,n, \\
W^{i-1}_{k,j} : \mbox{hidden layer I 에 있는 neuron k와 hidden layer +1에 있는 neuron j 사이의 weight} \\
n_i : i^{th} \mbox{hidden layer에 있는 neurons 수}
$$
hidden layer 둘 다 전부 네트워크에서 가중치를 초기화하기 위한 *kernel initializer*로 *uniform distribution*를 사용한다. 또한, multiple dimensions의 nonlinear datasets에서 여러 장점 때문에 활성화 함수로 ReLU(Rectified Linear Unit)를 선택했다. ReLU는 다음과 같이 주어진다.
$$
y = max(0,x)
$$
output layer는 y1, y2, y3의 세 레이어으로 구성됩니다.
네트워크 output은 다음과 같이 계산된다.
$$
y_i = f\Bigg(\sum^{n_N}_{k=1}W^N_{k,j}h^k_N \Bigg) \\

W^N_{k,j} : N^{th}\mbox{hidden layer에 있는 neuron k와 output layer에 있는 neuron j 사이의 weight} \\
n_N : N^{th} \mbox{hidden layer에 있는 뉴런의 수}
$$
output layer는 kernel initializer로 uniform distribution을 사용하지만, 히든 레이어와 달리 *Softmax*를 *activation function*으로 사용하여 logits을 probabilities(확률)로 나타낸다. Softmax function은 다음과 같다.
$$
F(X_i) = \frac{exp(X_i) i=0,1,2,...,k}{\sum^k_{j=0}exp(x_j)}
$$
  classification의 특징 때문에, equation (3)에 주어진 loss function으로 categorical crossentropy를 사용했다. 여기서 \hat y는 predict output 이다.
$$
L(y,\hat y) = -\sum^M_{j=0}\sum^N_{i=0}\Big(y_{ij} * log\Big(\hat y_{ij}\Big)\Big)
$$
Categorical crossentropy는 예측 분포(output layer 내 activations, 각 클래스에 하나씩)를 실제 분포와 비교한다. 여기서 true 클래스의 확률은 다른 클래스에 대해 1과 0으로 설정된다. 다른 많은 optimizers들 중에서, 우리는 제안된 모델을 최적화하기 위해 Adam (Adaptive Moment Estimation)을 사용했다. Adam은 각 파라미터에 대해 적응형 학습을 사용하며, 학습 속도의 가중치를 최근 gradients의 실행 평균으로 나눈다. 마지막으로, 모델은 배치 크기가 5이고 200개의 에포크(Epoch)로 교육할 수 있다. Table 1은 MLP를  fault classifier로 구성하는 데 사용되는 다양한 파라미터를 보여준다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.30.58 AM.png" alt="Screen Shot 2021-07-03 at 1.30.58 AM" style="zoom:50%;" />



 Bias-variance tradeoff를 확인하기 위해, *k-fold crossvalidation*은 5개의 validation을 training 데이터로 분할하여 수행한다. 또한 모델을 개선하고 오버핏을 줄이기 위해 드롭아웃 정규화 기법을 구현하였다. 첫 번째와 두 번째 hidden layer에 각각 0.1과 0.2의 dropout rates가 선택되었다. 모델의 평가, 개선 및 조정 결과는 섹션 4.1에 제시되어 있다.

![Screen Shot 2021-07-03 at 1.35.47 AM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.35.47 AM.png)

## Result and Discussions

### 4.1 Experimental Results

 실험 데이터 수집을 위해 PV 시스템에서 하드웨어 또는 회로 수정(circuit modification)이 없는 데이터는 "normal(정상)"으로 분류되었다. 고장 데이터는 PV 어레이 회로에서 몇 가지 의도적인 고장을 일으켜 수집되었다. Table 2는 제안된 모델을 교육하기 위해 다양한 환경 조건에서 수집된 데이터의 최소 범위, 최대 범위 및 편차(variance)를 보여준다. 실험 검증을 위해 Table 3과 Figure 6에 제시된 사양으로 실제 전력 생산 산업에 사용되는 PV 시스템을 설정했다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.37.07 AM.png" alt="Screen Shot 2021-07-03 at 1.37.07 AM" style="zoom:50%;" /><img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.37.23 AM.png" alt="Screen Shot 2021-07-03 at 1.37.23 AM" style="zoom:40%;" />

  Table 2와 같이, 겨울 데이터 세트의 변화(variance)는 정확한 예측을 위해 모델을 훈련하는 동안 각별한 주의가 필요한 여름 시즌보다 매우 크다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.40.43 AM.png" alt="Screen Shot 2021-07-03 at 1.40.43 AM" style="zoom:28%;" />   <img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.41.00 AM.png" alt="Screen Shot 2021-07-03 at 1.41.00 AM" style="zoom:26%;" />

  Figures 7,8은 각각 sunny days와 cloudy days 동안의 여름과 겨울의 input variable x1,x2의 normal 및 line-line fault dataset features를 보여준다.

   입력 데이터 집합 중에서  irradiation level은 절대 조건에서 가장 높은 분산 수준을 갖는 것으로 보인다. 그러나 현재 센서(S_I1/S_I2) 데이터를 시각화하는 것은 relative variance가 다른 input variables 중에서 가장 높았기 때문에 타당할 것이다. Figure 8에서 볼 수 있듯이, 겨울철에는 대부분 'normal cloudy'과 'line-line' fault data를 구별하기가 어렵다.

  Figure 9는 7-차원의 데이터(x1 ~ x7) 를 스케일링된 2차원 데이터로 시각화하는 principal component analysis (PCA) dimensional reduction technique를 보여준다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.49.15 AM.png" alt="Screen Shot 2021-07-03 at 1.49.15 AM" style="zoom:50%;" />

그림의 왼쪽 중앙과 오른쪽 중앙 부분에서 볼 수 있듯이, "normal" 상태 데이터와 "line-line" 결함 데이터(fault data)가 겹치는 작은 영역이 존재한다.

![Screen Shot 2021-07-03 at 1.50.54 AM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.50.54 AM.png)

  그림 10은 8:2의 train-test split 비율을 가진 데이터 세트의 training-test validation을 보여준다. 제안된 PNN 모델은 3000개의 데이터셋으로 광범위하게 훈련되었으며, 각 데이터셋은 PV  세스템의 서로다른 상태에서 1000개씩이었다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.52.47 AM.png" alt="Screen Shot 2021-07-03 at 1.52.47 AM" style="zoom:50%;" />

  Figure 11은 테스트 데이터를 교육 받은 모델에 가져올 때 100% 정확도의 결과를 제공하는 confusion matrix 를 보여준다. Confusion matrix에 표시된 수치 값은 절대 용어로 표현된다. 즉, line-line fault에 대한 총 예측 라벨 수는 184개였고, line-line fault에 대한 실제 라벨은 184개로 100%이다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.55.36 AM.png" alt="Screen Shot 2021-07-03 at 1.55.36 AM" style="zoom:50%;" />  

Figure 12 전북대학교 캠퍼스에서 실험용 1.8 kW PV 시스템의 설정을 보여준다.

![Screen Shot 2021-07-03 at 1.56.52 AM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.56.52 AM.png)

![Screen Shot 2021-07-03 at 1.56.34 AM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 1.56.34 AM.png)

  Table 4는 제안된 방법과 문헌에서 발견된 기존 연구의 비교를 보여준다. Figure 13은 fault detection을 위해 제안된 모델을 구현하는 개발된 데스크톱 애플리케이션의 스크린샷을 보여준다.

![Screen Shot 2021-07-03 at 2.54.26 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 2.54.26 PM.png)

### 4.2. Discussions

ML기반 fault detection과 진단 기술이 최근에 채택되었고, 향후 몇 년 동안 계속될 것으로 예상된다. ML기반 모델의 퀄리티는 training data에 크게 좌우된다. 연구에 따르면 PV 데이터로 교육받은 모델의 예측 정확도가 최대 100%에 이를 정도로 매우 높은 것으로 나타났다. 우리는 다른 기계 학습 모델을 사용하여 데이터셋을 테스트했고 Table 5에 나온 것처럼 각 사례에서 매우 높은 정확도(F1 점수)를 얻었다. 각  predicted classifier 사이의 상관관계는 그림 14와 같다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 2.54.46 PM.png" alt="Screen Shot 2021-07-03 at 2.54.46 PM" style="zoom:50%;" />



  ML Model을 트레이닝 하는 동안 인풋으로부터 주요 features를 찾아내는 것은 매우 중요하다. PV system에서 가장 중요한 features는 current와voltage이다. 이러한 두 가지 입력 기능으로만 교육된 고장 감지 모델은 더 많은 입력 데이터셋으로 교육된 다른 모델과 마찬가지로 강력할 수 있다. PV 시스템에서 모든 유형의 고장을 감지, 진단 및 찾을 수 있는 단일 고장 감지 기술은 없다. 서론 파트에서 논의된 바와 같이, 본 연구는 중대한 전기적 고장을 감지하는 것으로 제한된다. 본 연구의 일환으로, 우리는 포괄적이고 완벽한 고장 감지 방법을 개발하기 위해 ML을 포함한 하이브리드 기술을 개발하기 위한 연구를 계속하는 것을 목표로 한다.

  대부분의 경우, PV 시스템을 위해 개발된 고장 감지 모델은 다른 PV 시스템에 구현될 수 없다. electrical parameters는 주로 다른 PV 시스템에서 다양하기 때문이다. 약간의 수정만으로 모든 PV 시스템에 구현할 수 있는 유연한 모델을 개발할 필요가 있다. 우리는는 데스크톱 애플리케이션을 개발하는 동안 모델을 최대한 유연하게 만들기 위해 특별한 고려를 했다. 실험에 사용된 소스 코드와 데이터는 저자의 GitHub 페이지에서 오픈 소스 프로젝트로 사용할 수 있다. 오픈 소스 커뮤니티는 해당 애플리케이션을 위해 모델을 사용하거나, 피드백을 제공하거나, 모델 전반의 개선에 기여할 수 있다.



## Conclusion

  PV 시스템은 다양한 faults과 failures의 영향을 받으며, 그러한 faults과 failures의 조기 고장 감지는 PV 시스템의 효율성과 안전에 매우 중요하다. ML 기반 고장 감지 모델은 데이터로 교육되며 매우 높은 정확도로 예측 결과를 제공한다. 그러나, PV 시트템을 위한 데이터 기반 고장 감지 모델은 특히 environmental parameters를 고려하지 않을 경우 잘못된 예측을 제공할 수 있다. 이 논문에서는 고장 유형을 정확하게 분류하기 위해 PNN을 기반으로 PV 어레이를 위한 지능형 고장 감지 모델을 개발했다. 이 모델은 여름과 겨울의 다양한 환경 조건에서 서로 다른 데이터 값을 포함하는 대규모 데이터 세트를 사용하여 훈련되었다. 실험 검증을 위해 1.8kW(300W 패널 6개, 병렬로 연결된 라인 2개, 직렬로 연결된 패널 3개)에서 다양한 고장 상태 및 정상 상태 데이터 세트가 그리드 연결 PV 시스템으로 수집된다. 