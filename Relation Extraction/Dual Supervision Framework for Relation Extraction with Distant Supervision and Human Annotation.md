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

