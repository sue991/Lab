# Dual Supervision Framework for Relation Extraction with Distant Supervision and Human Annotation

## Abstract

RE는 knowledge base construction 및 question answering과 같은 실제 응용 분야에서 그 중요성으로 인해 광범위하게 연구되어 왔다. 

기존 작업의 대부분은 distantly supervised data나 human-annotated data에 대해 모델을 train한다.

Human annotation의 높은 정확성과 distant supervision의 저렴한 비용을 활용하기 위해 두 가지 유형의 데이터를 효과적으로 활용하는 이중 감독 프레임워크(dual supervision framework)를 제안한다.

그러나, RE model을 train하기 위해 간단하게 데이터의 두 타입을 결합하는 것은 distant supervision이 labeling bias가 있기 때문에 예측 정확도가 떨어질 수 있다.

우리는 각각 human annotation과 distant supervision에 의해 label을 예측하기 위해 *HA-Net*과 *DS-Net* 두개의 별도 예측 네트워크를 사용하여 distant supervision에 대한 잘못된 라벨링으로 인한 정확도 저하를 방지한다.

게다가, *HA-Net*이 distantly supervised labels로부터 배울 수 있게끔 *disagreement penalty*라고 불리는 추가적인 loss 용어를 제안한다.

추가적으로, contextual information을 고려하여 labeling bias를 적응적으로(adaptively) 평가하기 위해 추가적인 네트워크를 사용한다.

Sentence-level과 document-level RE에서의 성능 연구는 dual supervision 효과를 확인한다.

## Introduction

RE는 널리 knowledge base construction, question answering, biomedical data mining 등과 같은 실제 분야에서 널리 쓰이고 있다.

Text에서 entity pair가 주어졌을 때, RE의 목표는 text에 표현된 entity 사이의 관계를 발견하는 것이다.

더 구체적으로, 우리는 text에서 `<e_h,r,e_t>`의 형태인 triples를 추출하는 것에 중점을 둔다.

이때, `e_h`는 head entity, `e_t`는 train entity 그리고 `r`은 엔티티 사이의 relationship이다.

RE를 위한 모델을 train하기 위해 text-triple paris의 형태로 라벨링된 많은 양의 training data가 필요하다.

비록 human annotation이 RE model을 train하는데 높은 퀄리티의 labels를 제공해주긴 하나, manual labeling은 비싸고 시간도 많이 들기 때문에 large-scale training data를 만들기는 어렵다.

Thus, Mintz et al. (2009)가 external knowledge base(KB)를 사용해서 large labeled data를 자동으로 생산하는 distant supervision을 제안했다. 

Head entity `e_h` 와 tail entity `e_t`가 있는 text의 경우, relation type `r`에 대한 triple `⟨eh, r, et⟩`가 KB에 존재할 때, relationship이 text에 표현되지 않더라도 distant supervision에서 `<e_h,r,e_t>` 라는 레이블을 생성한다.

따라서 wrong labeling problem을 겪는다.

예를들어, triple `⟨U K, capital, London⟩`가 KB에 있으면, distant supervision은 심지어 `‘London is the largest city of the UK’` 이런 문장에서도 triple을 라벨링한다.

비록 두 라벨링 방법 각각 특정한 약점이 있지만, 대부분의 기존 RE 작접은 human-annotated(HA) data 또는 distant supervised(DS) data를 활용한다,

Human annotation의 높은 정확도와 distant supervision의 저렴한 비용을 활용하기 위해 적은 양의 HA 데이터 뿐만 아니라 많은 DS데이터도 효과적으로 활용할 것을 제안한다.

DS data는 *labeling bias*가 있을 가능성이 높기 때문에, 두 유형의 데이터를 결합하여 RE model을 train하기만 하면 예측 정확도가 저하될 수 있다.

Labeling bias를 자세히 살펴보려면 relation type의 *inflation*이 DS data와 HA data 각각의 text당 relation type의 평균 빈도의 비율이 되도록 한다.

만약 DS data에서 relation type의 평균 빈도가 HA data에서랑 똑같다면, 우리는 relation type이 *unbiased* 하다고 말한다(즉, relation의 inflation은 1). 

96개의 relation type을 가진 document-level RE dataset(DocRED)를 조사하여 relation type의 inflation이 0.48에서 85.9 사이임을 확인했다.

이것은 distant supervision이 몇몇 relation type에서 많은 양의 false labels를 생성하는 경향이 있음을 암시한다.

  최근에 Ye et al. (2019) 은 RE에 대한 labeling bias problem을 다루기 위해 domain adaptation approach를 도입했다.

이것은 DS 데이터에 대한 RE 모델을 train하고 HA 데이터를 사용하여 output layer의 bias term을 조정한다.

비록 bias 조절이 의미있는 정확도 향상을 성취할지라도, 한계가 있다.

이 방법의 기본적인 가정은 모든 텍스트에 대해 labeling bias가 static하다는 것이다. 왜냐하면 bias term은 train 후에 한 번만 조정되고 테스트 동안 동일한 bias을 사용하기 때문이다. 

그러나, labeling bias는 contextual information에 따라 다르다. 따라서 labeling bias를 고려하여 relation을 보다 정확하게 추출하기 위한 contextual information을 고려해야 한다.

  RE model을 training하는데 효과적으로 DS와 HA Data를 활용하기 위해, 추가적인 정확도 향상을 위해 대부분의 기존 RE model에 적용할 수 있는 *dual supervision framework*를 제안한다.

HA와 DS data에서 label distribution이 매우 다르기 때문에, 우리는 두 데이터 모두 multi-task learning problem으로 하여 RE model을 train하는 과제를 캐스팅했다.

따라서 우리는 두 개의 별도 output modules *HA-Net*과 *DS-Net*을 사용하여 각각 human annotation과 distant supervision에 의해 label을 예측하고, 이전 연구에서는 single output module을 활용한다.

이것은 human annotation과 distant supervision에 대한 label을 서로 다르게 예측할 수 있으므로 DS data의 잘못된 label로 인해 정확도가 저하되는 것을 방지할 수 있다.

만약  multi-task learning을 적용하기 위해 간단하게 prediction network를 분리한다면, *HA-Net*은 distant supervision labels로부터 학습할 수 없다.

DS data로부터 *HA-Net*을 학습할 수 있게 하기 위해, 추가적인 *disagreement penalty*라 불리는 loss term을 제안한다.

이것은 log-normal distribution과 maximum likelihood estimation을 사용하여 *HA-Net*을 업데이트하여 distant supervised labels를 효과적으로 반영하도록 HA-Net을 업데이트함으로써 예측 네트워크 *HA-Net*와 *DS-Net*로부터 output probability의 비율을 모델링한다.

게다가, 우리의 framework는 두가지 추가적인 네트워크인  μ-Net과 σ-Net을 이용하여 contextual information를 고려한 로그 정규 분포를 적응적으로(adaptive하게) 추정한다.

또한 이론적으로 disagreement penalty를 통해 HA-Net이 distant supervision에 의해 생성된 label을 효과적으로 활용할 수 있음을 보여준다.

마지막으로, 문장 수준 및 문서 수준 RE라는 두 가지 유형의 작업에 대한 dual supervision framework의 효과를 검증한다.

실험 결과는 우리의 dual supervision framework가 기존 RE model의 예측 정확도를 상당히 향상 시켰음을 보여준다. 

추가적으로. dual supervision framework는 sencent-level과 document-level RE 모두에서 상대적인 F1 score를 최대 32% 향상으로 최첨단 방법(Ye et al., 2019)을 크게 능가한다.

## Preliminaries

우리는 문장 수준 및 문서 수준 관계 추출의 문제를 제시하고 다음으로 관계 추출을 위한 기존 작업을 소개한다.

### 1. Problem Statement

(Yao et al., 2019; Wang et al., 2019) 논문에 따르면, 우리는 각 text에 entity mention이 추가된다고 가정한다.

Entity pair를 위해 문장은 보통 하나의 relationship을 설명하기 때문에, sentence-level relation extraction은 일반적으로 *multi-class* classification problem 으로 간주된다.

**Definition 2.1 (Sentence-level relation extraction)**  *head와 tail entity e_h, e_t의 pair, relation type set R 그리고 entity mentions에 annotated된 한 문장 s에 대해, 우리는 문장에서 e_h, e_t사이에서의 relation r ∈ R 을 결정한다. R이 e_h와 e_t 사이에 어느 relation도 존재하지 않음을 의미하는 special relation type `NA`를 포함한다는 것에 주목하자.*

  엔티티 쌍 사이에 있는 multiple relationship이 document에 표현될 수 있기 때문에, document-level relation extraction 보통 *multi-label* classification problem으로 정의된다.

**Definition 2.2 (Document-level relation extraction)**  *head와 tail entity e_h, e_t의 pair, relation type set R 그리고  entity mentions에 annotated된 한 document d에 대해, 우리는 e_h 와 e_t 사이의 모든 relation의 집합 R\* ⊂ R이 document d에서 나타난다는 것을 발견했다. R\*의 empty set에 의해 표현될 수 있기 때문에 R이 `NA`를 포함하지 않는 것에 주목하자.*

  이 논문에서, 우리는 주로 sentence level RE에 대해 논의하고 document-level RE로 framework를 확장한다.

![Screen Shot 2021-08-21 at 6.47.41 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-21 at 6.47.41 PM.png)

### 2. Existing Works of Relation Extraction

전형적인 RE model은 Figure 1(a)에서 보이는 것 처럼 feature encoder와 prediction network로 구성된다.

Feature encoder는 text를 head와 tail 엔티티의 hidden representation으로 변환한다.

Cai et al. (2016)과 Wang et al. (2019)은 각각 Bi-LSTM과 BERT를 이용하여 text를 인코딩한다.

반면에, Zeng et al. (2014)과 Zeng et al. (2015) 은 CNN을 인코더로 사용한다.

추가적으로,  Zeng et al. (2014)은 각 단어에서 head와 tail entity까지의 relative distance를 고려하는 position embedding을 제안한다.

  Prediction network는 entity 사이의 relation의 probability distribution을 아웃풋한다.

Sentence-level RE이 multi-class classification task이기 때문에, sentence-level RE models는 *softmax classifier*를 prediction network로써 활용하고 categorical cross entropy를 loss function으로써 사용한다.

반면 document-level RE models는 *sigmoid classifier*와 binary cross entropy를 각각 prediction network과 loss function 으로 사용한다.

Distant supervision으로부터 획득된 labels가 single prediction network에서는 noisy하고 biased하기 때문에, DS data와 HA data를 함께 정확하게 예측하는 것은 어렵다.

## Dual Supervision Framework

우리는 먼저 효과적으로 HA와 DS data 모두 RE models을 training하는데 활용하기 위한 dual supervision framework의 개요를 보여준다.

다음 framework에서 output layer의 자세한 구조를 보여주고 distant supervision의 labeling bias를 고려하는 disagreement penalty와 함께 새로운 loss functiond을 제안한다.

그다음 어떻게 제안된 모델을 두 타입의 데이터로 잘 훈련하는지 방법과 test data에서 relation을 추출하는 방법을 설명한다.

마지막으로, 우리는 별도의 prediction network를 사용하지만 어떻게 disagreement penalty로 인해 각 prediction network가 다른 prediction network의 label로부터 학습하는지에 대해 논의한다.

### 1. An Overview of the Dual Supervision Framework

Figure 1(b)를 보면, 우리의 framework는 4개의 sub-networks와 함께 feature encoder와 output layer로 구성된다.

이것은 기존 RE 모델의 정확도를 높이기 위해 다양한 RE 모델을 수용할 수 있을 만큼 일반적이다.

우리의 framework를 model의 feature encoder를 사용하고 original prediction network의 구조를 이용한 4개의 sub-network를 만들어 기존 RE model에 적용할 수 있다.

우리의 framework가 기존 모델의  feature encoder를 사용하기 때문에, 간단히 output layer에 대한 내용만 설명한다.

  이전 연구와 달리, human annotated labels과 distantly supervised labels의 예측에서의 차이를 허용하기 위해 HA-Net과 DS-Net이라는 두 개의 개별 예측 네트워크를 사용하여 각각 HA data와 DS data의 labels를 예측함으로써 multi-task learning을 활용한다.

우리는 또한 HA-Net을 text data로부터 relation을 추출하는데 사용한다.

Prediction networks의 분리는 distant supervision의 부정확한 labels에 의한 정확도 감소를 예방한다.

만약 간단하게 두 개의 prediction networks를 multi-task learning에 적용하도록 활용한다면, 비록 prediction network가 feature encoder를 공유할지라도 *HA-Net*은  distantly supervised labels로부터 학습할 수 없다.

*HA-Net*이 distantly supervised labels로부터 학습할 수 있도록 하기 위해, 우리는 *disagreement penalty*라고 불리는 추가적인 loss term을 소개한다.

이것은 log-normal distributions와 maximum likelihood estimation를 이용하여 HA-Net과 DS-Net의 output 사이의 disagreement를 모델링한다.

게다가 contextual information을 고려하면서 log-normal distribution의 파라미터를 adaptive하게 추정하기 위해 , 두개의 parameter networks μ-Net과 σ-Net을 이용했다.

  Label `⟨eh, r, et⟩`에서, I_HA를 human annotation에 의해 획득된 label일 경우 1, 아니면 0인 indicator variable이라 하자.

제안된 framework는 label `⟨eh, r, et⟩` 에서 다음과 같은 loss function을 사용한다.
$$
L_{h,t} = I_{HA}·L^{HA}_{h,t} + (1-I_{HA})·L^{DS}_{h,t} + \lambda·L^{DS-HA}_{h,t} \\

L^{HA}_{h,t} , L^{DS}_{h,t} : \mbox{HA-Net and DS-Net의 prediction loss} \\
L^{DS-HA}_{h,t} : \mbox{HA-Net, DS-Net에 의한 prediction 사이의 distance를 캡처하기 위한 disagreement penalty} \\
$$
Hyperparameter `λ`는 prediction error에 대한 disagreement panalty의 상대적 중요성(relative importance)를 컨트롤한다.

각 타입의 데이터에 대해 다른 prediction network를 사용하고 disagreement penalty를 도입함으로써, HA-Net은 noisy DS data에 overfitting 하는것을 줄이면서 distantly supervised labels로부터 학습한다.

### 2. Separate Prediction Networks

DS data에서 noisy labels로부터의 정확도 감소를 완화하기 위해, 두 개의 prediction network를 사용한다.

*HA-Net* network는 train data로부터 human-annotated labels를 예측하는데 사용되고, test data로부터 relations을 예측하는데 사용된다. 다른 prediction network *DS-Net*은 distant supervision에 의해 획득된 label을 예측하는데 사용된다.

우리는 모델 파라미터를 공유하지 않고 우리의 framework의 두 prediction networks에 대해 기존 모델의 prediction network를 사용한다.

prediction networks *HA-Net* 과 *DS-Net*은 |R|-dimensional vector를 아웃풋한다.
$$
p^{HA} = [p(r_1|e_h,e_t,HA), ... , p(r_{|R|}|e_h,e_t,HA)] \\
p^{DS} = [p(r_1|e_h,e_t,DS), ... , p(r_{|R|}|e_h,e_t,DS)] \\

p(r|e_h,e_t,HA), p(r|e_h,e_t,DS) : \mbox{DS, HA data에 있는 label ⟨eh,r,et⟩의 probabilities} \\
\mbox{두 개를 각각 }p^{HA}_r , p^{DS}_r \mbox{라고 정의}
$$

### 3. Disagreement Penalty

Distant supervision labels는 biased되었고 bias의 사이즈는 relation의 타입에 따라 다르다.

게다가, bias는 text의 content뿐만 아니라 head와 tail entity의 타입과 같은 많은 다른 features에 따라 달라질 수 있다.

따라서 우리는 head와 tail entity 가 위치한 context에 따라 labeling bias를 모델링하기 위해 효과적인 disagreement penalty를 사용할 것을 제안한다.

**Distribution of inflations.**   우리는 labeling bias를 relation의 inflation을 사용하여 labeling bias를 측정한다. Relation type의 inflation은 DS Data와 HA Data의 text당 relation type의 평균 frequency의 비율이다.

Inflations의 분포를 조사하기 위해, 우리는 DocRED data에 있는 96개의 relation types의 inflation을 계산했다.

Kolmogorov-Smirnov (K-S) test (Massey Jr, 1951)가 관찰된 데이터가 주어진 확률 분포에서 추출되는지 여부를 결정하는 데 널리 사용되기 때문에, 이 분포를 사용하여 inflation의 최적의 분포를 찾았다.

Inflation의 범위가 [0,∞)이기 때문에, [0,∞)에서 지원되는 4개의 확률분포(Log-normal, Weibull, chi-square 그리고 exponential distributions)의 p-values을 평가했다.

게다가, 우리는 normal distribution을 baseline으로 포함한다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-08-22 at 11.07.19 PM.png" alt="Screen Shot 2021-08-22 at 11.07.19 PM" style="zoom:50%;" />

Table 1은 DocRED Data에서 K-S test의 결과를 보여준다.

Probability distribution이 data에 잘 맞는다면, 높은  p-value값을 가지는 것에 주목하자.

Log-normal distribution이 가장 높은 p-value를 갖기 때문에, 이 분포는 5개의 probability distribution 중 가장 적합도가 높은 distribution이다.

관찰 결과에 근거하여, 우리는 두 prediction networks의 output 사이에서 disagreement penalty를 모델링한다.

**Modeling the disagreement penalty.**   우리는 maximum likelihood estimation에 근거한 disagreement penalty를 개발한다.  Xr을 pr_DS와 pr_HA의 비율을 나타내는 random variable이라고 가정하자.

Inflation이 DS data와 HA data에서의 labels의 수의 비율이기 때문에, pr_DS/pr_HA 비율은 relation type `r`의 *conditional inflation*을 표현한다.

따라서 Xr이  probability density function이 다음과 같은 log-normal distribution L(μr , σr2)을 따르는 것을 가정한다:
$$
f(x) = \frac{1}{x\sigma_r\sqrt{2\pi}}exp\bigg(-\frac{(\log x - μ_r)^2}{2\sigma^2_r}\bigg)
$$
Disagreement penalty L^{DS-HA}_{h,t}는 conditional inflation pr_DS/pr_HA의 negative log likelihood로 정의되는데, 이것은 Equation (2)에서 pDS/pHA을 대체한다:
$$
-\log f (p^{DS}_r/p_r^{HA}) = \frac{1}{2} \bigg( \frac{\log p_r^{DS} - \log p_r^{HA} - μ_r}{\sigma_r} \bigg)^2 + \log p^{DS}_r - \log p_r^{HA} + \log \sigma_r + \frac{\log2\pi}{2}
$$
log 2π/2가 contant이기 때문에, constant term없이 Equation (3)에서  disagreement penalty를 활용한다.

만약 μr과 σr 값을 고정시킨다면, context마다 매우 다르기 때문에 우리는 효과적으로 conditional inflation을 평가할 수 없다.

예를 들어, relation type `capital` 의 inflation이 매우 높을지라도, ‘is the capital city of’과 같은 특정한 구가 text에 나타나면 conditional inflation은 낮아야 한다.

Contextual information을 고려하기 위해, 우리는 log-normal distribution L(μr, σr2)의 파라미터인 μr과 σr을 추정하기 위해 두 개의 추가적인 네트워크 μ-Net과 σ-Net을 사용한다.

### 4. Parameter Networks

파라미터 네트워크 μ-Net과 σ-Net은 벡터 `μ = [μ1, ..., μ|R|]` 과 `σ = [σ1, ..., σ|R|]`을 아웃풋하는데, 이것들은 r ∈ R의 conditional inflation을 표현하기 위한 log-normal distributions의 파라미터이다.

μ-Net과 σ-Net은 모두 output activation function을 제외하고 예측 네트워크의 것과 동일한 구조를 가진다.

Log-normal distribution L(μ, σ)의 경우, parameter μ양수 또는 음수일 수 있으며, σ는 항상 양수이다.

따라서 우리는  hyperbolic tangent function과 softplus function을 μ-Net과 σ-Net 각각의 output activation function으로 사용한다.

  예를 들어, 만약 original RE model의 prediction network는 bilinear layer와 output activation function으로 구성되었다면, 파라미터 벡터  μ ∈ R^|R|와 σ ∈ R^|R| 는 head entity vector h∈R^d와 tail entity vector t∈R^d 로 계산된다.
$$
μ = tanh(h^⊤W^μt + b^μ), \\ σ = softplus(h^⊤W^σt + b^σ) + ε,\\

softplus(x) = log(1+e^x)
$$
ε는 극히 작은 σ_r 값이 손실 함수를 지배하지 못하도록 하는 sanity bound이고,

W^μ ∈ R^{d×|R|×d} , W^σ ∈ R^{d×|R|×d}, b^μ ∈ R^|R| , b^σ ∈ R^|R|은 학습가능한 파라미터 이다.

sanity bound ε은 0.0001로 설정하였다.

### 5. Loss function

Sentence-level relation extraction에서, categorical cross entropy loss를  prediction losses인 L^HA\_{h,t} 와 L^DS\_{h,t}로 사용한다.

Label `⟨eh,r,et⟩`을 위해, Equations (1)과 (3)에서 다음과 같은 손실 함수를 구한다.
$$
L_{h,t} = I_{HA} · L^{HA}_{h,t} + (1-I_{HA}) · L^{DS}_{h,t} + \lambda·L^{DS-HA}_{h,t} \\
= -I_{HA}·\log p^{HA}_r - (1-I_{HA}) \log p_r^{DS} + \lambda \bigg[ \frac{1}{2} \bigg( \frac{l_r-μ_r}{\sigma_r} \bigg)^2 + l_r + \log \sigma_r \bigg] \\

l_r =\log p^{DS}_r − \log p^{HA}_r , I_{HA} : \mbox{HA data의 label이면 1, 아니면 0}
$$

### 6. Analysis of the Disagreement Penalty

w_HA가 test에서 relation을 예측하는 HA-Net의 learnable parameter라고 하자.

우리는 human annotated label과 distantly supervised label에 대한 w_HA에 관련하여 loss function의 gradients를 비교함으로써 disagreement lenalty의 효과를 조사한다,

Label `⟨e_h ,r,e_t⟩` 에 대해,
$$
\phi_r = (\log (p^{DS}_r / p^{HA}_r) - μ_r)/\sigma^2_r \mbox{라고 하자.}
$$
만약 레이블이 human annotated이면, Equation (4)로부터 w_HA에 관한 loss L_{h,t}의 다음과 같은 gradient를 얻는다.
$$
∇ L_{h,t} = ∇L^{HA}_{h,t} + 0 + \lambda ∇L^{DS-HA}_{h,t} = -(1+\lambda(1+\phi_r)) \frac{1}{{p_r}^{HA}}∇p_r^{HA}
$$
반면, 만약 label이 distant supervision에 의해 annotated되었으면, gradient는 다음과 같다:
$$
∇L_{h,t} = 0+0+\lambda∇L^{DS-HA}_{h,t} = -\lambda(1+\phi_r)\frac{1}{p_r^{HA}}∇p_r^{HA}
$$
Equations (5) and (6)에 나온 두 gradients는 같은 `−∇p^HA`의 direction을 갖는다.

이것은 human annotated label과 distantly supervised label이 각각 `1+λ(1+φr)`, `λ(1+φr)`로 교정된다는 점을 제외하고 *HA-Net training에 비슷한 영향을 미친다는 것을 의미한다.

따라서, HA-Net은 disagreement penalty를 도입함으로써 human annotated labels뿐만 아니라 distantly supervised labels로부터 학습할 수 있다.

Log-normal distribution L(μr,σr)은 head entity와 tail entity가 있는 주어진 문장에 대한 conditional inflation을 설명한다.

만약 L(μr,σr) 의 median e^{μr}이 높은 값을 가지고 있다면, distantly supercised label은 false label이다.

따라서, 우리는 distantly supervised label의 영향을 줄이기 위해 φr의 사이즈를 줄인다.

반면에, median e^{μr}이 낮아진다면, φr의 사이즈는 적극적으로 distantly supervised label을 활용하기 위해 증가한다.

### 7 Extension to Document-level Relation Extraction

Document-level RE에서,  prediction losses L^{HA}와 L^{DS}로써 *binary* cross entropy를 사용한다.

e_h,e_t의 entity pair 에서, R_{h,t}를 entities 사이의 relation types의 집합이라고 하자.

Train에서, 우리는 document RE에서 다음과 같은 loss function을 사용한다:
$$
L_{h,t} = -I_{HA}\Bigg(\sum_{r\in R_{h,t}}{\log p_r^{HA}} + \sum_{r \in R/R_{h,t}}\log(1- p_r^{HA}) \Bigg) \\ - (1-I_{HA})\Bigg( \sum_{r\in R_{h,t}} \log p_r^{DS} + \sum_{r\in R/R_{h,t}} \log (1-p_r^{DS}) \Bigg) + \lambda \sum_{r \in R_{h,t}}\Bigg[ \frac{1}{2} \bigg( \frac{l_r-μ_r}{\sigma_r} \bigg)^2 + l_r + \log \sigma_r \Bigg] \\

l_r = \log p_r^{DS}/p_r^{HA}, I_{HA} : \mbox{HA Data로부터 label이면 1, 아니면 0}
$$
우리는 위의 loss function에 대해 Section 3.6에서 보여준 것과 같은 property를 얻는다.

Test에서, 우리는 만약 pr^{HA}가 dev set에서 조율된 threshold보다 크다면 모델이 triple`⟨e_h, r, e_t⟩`을 아웃풋한다고 간주한다.

## Experiments

(Ye et al., 2019)과 (Yao et al., 2019; Wang et al., 2019)의 실험 설정을 따라 문장 수준 및 문서 수준 RE에 대한 성능 연구를 실시했다.

모든 모델은 PyTorch 와 V100 GPU에서 수행되고 훈련된다.

우리는 *HA-Net* 과 *DS-Net*을 같은 initial parameter를 갖도록 initialized했다.

구현을 포함한 자세한 실험 정보는 부록 A에서 확인할 수 있다.

### 1. Experimental Settings

**Dataset.**   KBP (Ling and Weld, 2012; Ellis, 2012) 와 NYT (Riedel et al., 2010; Hoffmann et al., 2011)이 sentence- level RE를 위한 데이터셋이고, DocRED가 document-level RE를 위한 데이터셋이다.

데이터셋들의 통계는 Table 2에 요약되어있다. KBP와 NYT 가 HA train data를 가지고 있지 않기 때문에, HA train data의 20%를 HA test data로 사용한다.

랜덤으로 KBP와 NYT의 train data의 10%를 dev data로 만들었다.

DocRED의 test data의 ground truth가 공개되어 있지 않다는 것에 주목하자.

그러나, 우리는 test data로부터 추출된 결과의 F1 score를 CodaLab에서 주최하는 DocRED competition에 제출하면 얻을 수 있다. (available at https://competitions.codalab.org/competitions/20717)

우리는 dev data와 test data로부터 계산된 F1 scores를 기록한다.

**Compared methods**   우리는 *DUAL*이라고 명시된 우리의 dual supervision framework를 최신 방법인 *BASet*과 *BAFix*와 비교한다.

Sentence-level RE에서, 우리는 *DUAL*과 Doc-level RE에서 사용될 수 없고 multi-class classification에서만 이용가능한 두 추가적인 baselines인 *MaxThres*와 *EntThres*와 비교한다.

*MaxThres*는 만일 maximum output probability가 threshold보다 작다면 `NA`를 아웃풋한다.

비슷하게도, *EntThres*는 output probability distribution의 entropy가 threshold보다 크다면 `NA`를 아웃풋한다.

**Used relation extraction models. **   *sentence-level RE*에서, 우리는 6개의 모델을 사용한다: * ***BiGRU***s* (Zhang et al., 2017), ***PaLSTM****s* (Zhang et al., 2017), ***BiLSTM****s* (Zhang et al., 2017), ***CNN****s* (Zeng et al., 2014), ***PCNN****s* (Zeng et al., 2015) and ***BERT****s* (Wang et al., 2019). On the other hand, for *document-level RE*, we used the five models: ***BERT****D* (Wang et al., 2019), ***CNN****D* (Zeng et al., 2014), ***LSTM****D* (Yao et al., 2019), ***BiLSTM****D* (Cai et al., 2016) and ***CA****D* (Sorokin and Gurevych, 2017).

*CNND*, *BiLSTMD*  그리고 *CAD*는 원래 sentence-level RE에 제안되었고, 우리는 document-level RE의 적용에 사용한다.

추가적으로, sigmoid 를 softmax로 output activation function을 바꿈으로써 *BERTD*를 sentence-level RE에 적용한다.

### 2. Comparison with Existing Methods

Dual supervision framework를 기존 방법과 비교한다.

**Sentence-level RE.**    Table 3은 KBP와 NYT에서 relation extraction을 위한 F1 scores 를 보여준다.

*DS-Only*와 *HA-Only*은 각각 distantly supervised와 human-annotated labels에 훈련된 original RE models를 표현한다. 

*DUAL*은 *BiLSTMs*을 제외하고 모든 RE models에서 가장 높은 F1 scores를 보여준다.

KBP와 NYT는 train data에 human-annotated labels 수가 적기 때문에 HA-Only는 DS-Only보다 F1 점수가 더 나쁘다.

게다가 *DUAL* 은 적은 양의 human-annotated labels를 추가로 사용하여 DS에 비해 5%에서 40%로 F1 score의 향상시킨다.

반면, 비교하는 방법인  *BAFix*, *BASet*, *MaxThres* and *EntThres*는 종종 *DS-Only* 랑 *HA-Only* 보다 낮은 성능을 보인다

**Document-level RE.**   Table 4에서 DocRED에 대한 F1 score를 보여준다.

DUAL은 모든 RE model 에서 *BASet*과 BAFix보다 높은 성능을 보여준다.

특히, BERTd를 쓰는 dual framework의 F1 score는 BASet과 BAFix보다 22% 이상의 향상을 보여준다.

DocRED가 큰 human-annotated train data를 가지고 있기 때문에, *HA-Only*는 *DS-Only*보다 좋은 성능을 보여준다.

*BERT**D* and *CNN**D*을 위해, 기존 방법은 HA-Only와 비교했을 때 낮은 성능을 보여준다.

