# Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling

https://github.com/wzhouad/ATLOP

## Abstract

 Document-level relation extraction (RE)은 sentence-level에 비해 새로운 과제를 제기한다. 일반적으로 하나의 문서에는 여러 개의 엔티티 쌍이 포함되어 있으며 하나의 엔티티 쌍은 여러 개의 가능한 relation과 연결된 문서에서 여러 번 발생한다. 이 논문에서, 우리는 multi-label과 multi-entity problems를 해결하기 위해 두개의 새로운 기술인 adaptive thresholding과 localized context pooling을 제안한다. Adaptive thresholding은  이전 작업의 multi-label classification에 대한 global threshold를 learnable entities-dependent threshold로 대체한다. Localized context pooling은 pre-trained language models에서 relation을 직접적으로 전달하여 relation을 결정하는 데 유용한 관련 컨텍스트를 찾는다. 우리는 세 가지  document-level RE benchmark datasets를 사용하여 실험한다: DocRED, CDR, GDA. 우리의 ATLOP(**A**daptive **T**hresholding and **L**ocalized c**O**ntext **P**ooling) 모델은 63.4의 F1 score를 달성했고, 또한 상당히 CDR과 GDA에 있는 모델보다 좋은 성능을 보였다. 

## Introduction

RE는 주어진 텍스트에서 두 엔티티 간의 관계를 식별하는 것을 목표로 하며 information extraction에 중요한 역할을 한다. 기존 연구는 주로 sentence-level relation extraction, 즉 단일 문장에서 엔티티 간의 관계를 예측하는 것에 초점을 맞춘다. 그러나, 위키백과 기사나 생물 의학 문헌의 관계적 사실과 같은 많은 양의 관계는 실제 적용에서 여러 문장으로 표현된다. 일반적으로 문서 수준 관계 추출이라고 하는 이 문제는 전체 문서에서 엔티티 간의 복잡한 상호 작용을 캡처할 수 있는 모델이 필요하다. 문장 수준의 RE와 비교하여 문서 수준의 RE는 고유한 과제를 제기한다. TACRED 및 SemEval 2010 Task 8과 같은 문장 수준의 RE 데이터셋의 경우, 문장에는 분류할 엔티티 쌍이 하나만 포함된다. 반면에 문서 레벨 RE의 경우 한 문서에 여러 엔티티 쌍이 포함되어 있으므로 이들 관계를 한 번에 분류해야 한다. RE 모델은 특정 엔티티 쌍과 관련된 컨텍스트를 사용하여 문서의 일부를 식별하고 초점을 맞추어야 한다. 또한, 문장 수준 RE의 경우 엔티티 쌍당 하나의 관계가 발생하는 것과 대조적으로 문서 수준 RE의 뚜렷한 관계와 관련된 문서에서 한 개의 엔티티 쌍이 여러 번 발생할 수 있다. 문서 수준 관계 추출의 이러한 multi-entity(문서에서 분류할 다중 엔티티 쌍) 및 multi-label(특정 엔티티 쌍에 대한 다중 관계 유형) 속성은 문장 수준 관계 추출보다 어렵다. Figure 1은 DocRED 데이터 세트의 예를 보여준다.

![Screen Shot 2021-07-12 at 8.49.12 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-12 at 8.49.12 PM.png)

이 Task는 엔티티 쌍(색상으로 강조 표시됨)의 relation types을 분류하는 것이다. 특정 개체 쌍(*John Stanistreet, Bendigo*)의 경우, 첫 두 문장과 마지막 문장에 의해 두 개의 relation *place of birth* 와  *place of death* 가 표현된다. 다른 문장은 이 엔티티 쌍에 대해 부적절한 정보를 포함하고 있다.

  multi-entity problem을 해결하기 위해 대부분의 최신 접근 방식은 dependency struc- tures, heuristics 또는 structured attention을 이용한 document graph를 생성한 다음  graph neural models을 사용하여 추론을 수행한다. Constructed graphs는 문서에서 멀리 흩어져 있는 엔티티를 브리지(bridge)하여 장거리 정보를 캡처하는 데 RNN 기반 encoders의 결점을 완화시킨다. 그러나, transformer-based models은 암시적으로 long-distance dependencies를 모델링할 수 있기 때문에, graph structures가 BERT와 같은 사전 교육 언어 모델에 도움이 되는지 여부는 불분명하다. 그래프 구조를 도입하지 않고도 사전 교육 언어 모델을 직접 적용할 수 있는 접근 방식도 있었다. 단순히 엔티티 토큰의 embedding을 평균하여 entity embeddings을 얻고 이를 classifier에 입력하여 relation labels을 가져온다. 그러나, 각 엔티티는 서로 다른 엔티티 쌍에서 같은 representation을 가지므로, 관련없는 context에서 noise가 발생할 수 있다.

  이 논문에서 graph structures를 사용하는 대신, localized context pooling를 제안하고 있다. 이 기술은 모든 엔티티 쌍에 대해 동일한 엔티티 임베딩 사용 문제를 해결한다. 이는 현재 엔티티 쌍과 관련된 추가 컨텍스트와 함께 entity embedding을 향상시킨다. 새로운 context attention layer를 처음부터 교육하는 대신, pre-trained language models에서 attention heads를 직접 전송(transfer)하여 entity-level attention를 얻는다. 그런 다음 쌍으로 구성된 두 엔티티에 대해, 곱셈(multiplication)을 통해 그들의 attention을 합쳐 해당 엔티티 모두의 중요한 컨텍스트를 찾는다.

  Multi-label problem의 경우, 기존 접근 방식은 이를 binary classification problem으로 줄인다. 트레이닝 후에, global threshold가 class probabilities에 적용되어 relation labels를 가져온다. 이 방법은 heuristic threshold tuning을 포함하고, development data로부터 조정된 threshold가 모든 인스턴스에 최적화된 것이 아닐 경우 decision errors를 발생시킨다.

  이 연구에서는 global threshold를 learnable threshold class로 대체하는 adaptive thresholding 기술을 제안한다. Threshold class는 adaptive-threshold loss로 학습되는데, 이것은 모델 트레이닝에서 positive class의 logit을 threshold 이상으로 밀어내고 negative class의 logit을 아래로 끌어당기는 *rank-based* loss이다. 테스트 시 threshold class 보다 로짓이 높은 클래스를  predicted labels로 return하거나, 해당 클래스가 없는 경우 NA를 리턴한다.  이 기술을 사용하면 threshold tuning이 필요하지 않을 뿐 아니라 서로 다른 엔티티 쌍에 맞춰 임계값을 조정할 수 있으므로 훨씬 더 나은 결과를 얻을 수 있다.

  제안된 두 기법을 결합하여, pre-trained language models의 힘을 최대한 활용하기 위해 **ATLOP**(Adaptive Thresholding and Localized cContext Pooling)라는 단순하면서도 효과적인 관계 추출 모델을 제안한다. 이 모델은 문서 수준 RE의 multi-label 및 multi-entity problems를 해결한다. 문서 수준 관계 추출 데이터셋인 DocRED (Yao et al. 2019), CDR (Li et al. 2016), GDA (Wu et al. 2019b)에 대한 실험은 ATLOP 모델이 최첨단 방법을 훨씬 능가함을 보여준다. 우리 연구의 기여는 다음과 같이 요약된다:

- 엔티티 쌍에 의존하는 adaptive threshold를 학습할 수 있도록 하고 global threshold를 사용함으로써 발생하는 decision errors를 줄이는 adaptive-thresholding loss를 제안한다.
- 우리는 localized context pooling을 제안하는데, 이것은 entity pairs가 더 좋은 entity representation을 얻기 위해 관련된 context를 잡기 위해 pre-trained attention을 transfer한다.

- 우리는 3개의 공개 문서 레벨 관계 추출 데이터셋에 대한 실험을 수행한다. 실험 결과는 3개의 벤치마크 데이터셋에서 최첨단 성능을 달성하는 ATLOP 모델의 효과를 보여준다.

## Problem Formulation

document *d*와 엔티티 셋 `{e_i}^n_{i=1}`이 주어졌을 때, doc-level RE의 task는 entity pairs (e_s, e_o)_{s, o=1...n; s != o} 사이에서 R ∪ {NA}의  subset relation을 예측하는 것이다. 이때 R 은 미리 정의된 interest relation set이고, e_s, e_o는 각각 subject, object 엔티티이다. 엔티티 e_i가 entity mentions 에 의해 document에 여러번 발생할 수 있다.
$$
\mbox{entity mentions : } \{m^i_j\}^{N_{e_i}}_{j=1}
$$
Mentions의 pair로 표현되는 경우, 엔티티 (es, eo) 사이에 relation이 존재한다. Relation을 표현하지 않는 엔티티 쌍은 NA 로 레이블링된다. 테스트 시, 모델은 문서 d에서 모든 엔티티 쌍(es, eo)_{s,o=1...n;s!=o}의 레이블을 예측해야 한다.

## Enhanced BERT Baseline

 이 섹션에서, 우리는 doc-level RE 에서의 우리의 기본 모델을 보여준다. 우리는 기존 BERT 기준을 기반으로 모델을 구축하고 다른 기술을 통합하여 성능을 더욱 향상한다.

### Encoder

document d = [x_t]^l_{t=1}이 주어졌을 때, mentions의 시작과 끝에 특수 기호 "*"를 삽입하여 엔티티 mention의 위치를 표시한다. 이것은 엔티티 마커 기법에서 적용된다. 그런 다음 문서를 pre-trained language model에 삽입하여 상황에 맞는 임베딩을 얻는다.
$$
H = [h_1,h_2,...,h_l] = BERT([x_1,x_2,...,x_l])
$$
이전 연구에 따르면, document는 인코더에 의해 한 번 인코딩되며, 모든 엔티티 쌍의 classification는 동일한 컨텍스트 임베딩을 기반으로 한다. mention의 시작에 있는 "*" embedding을 mention embeddings로 간주한다. 한 mentions {mij}^N_{j=1} 에서 엔티티 ei의 경우, maxpooling의 smooth 버전인 logsumexp pooling을 적용하고, entity embedding h_ei를 얻는다.
$$
h_{e_i} = log\sum^{N_{e_i}}_{j=1}exp\Big(h_{m^i_j}\Big)
$$
이 풀링은 문서에 언급된 신호를 누적한다. 이것은 실험에서의 평균 풀링에 비해 성능이 우수하다.

### Binary Classifier

엔티티 쌍 `es,eo` 의 embedding (h_es, h_eo)가 주어졌을때, 선형 레이어에 이어 non-linear activation으로 엔티티를 hidden states z에 매핑한 다음, bilinear function 및 sigmoid activation으로 relation r의 확률을 계산한다. 이 과정은 다음과 같다.
$$
z_s = tanh(W_sh_{e_s}), \\ 
z_o = tanh(W_oh_{e_o}), \\
P(r|e_s,e_o) = \sigma(z^\intercal_sW_rz_o /= b_r), \\
\mbox{model parameters : } W_s ∈ R^{d*d} , W_o ∈ R^{d*d}, W_r ∈ R^{d*d}, b_r ∈ R
$$
한 엔티티의 representation은 서로 다른 엔티티 쌍 간에 동일하다. Bilinear classifier에 있는 파라미터 수를 줄이기 위해, group bilinear를 사용하는데, 이것은 embedding dimensions를 k개의 동일한 크기의 그룹으로 나누고 그룹 내에서 bilinear를 적용한다.
$$
[z^1_s;...;z^k_s] = z_s, \\
[z^1_o;...;z^k_o] = z_o, \\
P(r|e_s,e_o) =\sigma \Bigg(\sum^k_{i=1}z^{i\intercal}_s W^i_r z^i_o + b_r \Bigg), \\
\mbox{model parameters : }W^i_r ∈ R^{d/k × d/k} \mbox{ for }i = 1..k  \\
P(r|e_s,e_o) : \mbox{relation r이 entity pair }(e_s,e_o) \mbox{ 와 관련된 확률}
$$
이 방법으로, 파라미터 수를 d^2에서 d^2/k로 줄일 수 있다. 우리는 트레이닝하는데  binary cross entropy loss를 사용한다. inference중에는 dev set의 evaluation metrics(RE의 경우 F1 score)를 최대화하는 global threshold θ를 조정하고 P(r|es,eo) > θ일 경우 관련된 relation으로 r을 리턴하거나, relation이 없을 경우 NA를 리턴한다.

  NAT의 향상된 기본 모델은 실험에서 기존 BERT 기준선을 훨씬 능가하는 거의 최신 성능을 달성한다.

## Adaptive Thresholding

RE Classifier는 relation labels로 변환되어야 하는 [0, 1] 범위 내의 확률 P(r|es, eo)를 출력한다. threshold에는 closed-form solution이 없으며 differentiable도 없기 때문에 threshold를 결정하는 일반적인 방법은 범위(0, 1)의 여러 값을 열거하고 evaluation metrics(RE의 경우 F1 score)을 최대화하는 값을 선택하는 것이다. 그러나 모델은 하나의 global threshold으로는 충분하지 않은 여러 엔티티 쌍 또는 클래스에 대해 서로 다른 신뢰도를 가질 수 있다. relations 수는 다양하며(multi-label problem), 모델은 글로벌하게 보정되지 않을 수 있으므로 동일한 확률이 모든 엔티티 쌍에 대해 동일하지 않다. 이 문제는 우리가 global threshold를 학습 가능하고 adaptive한 threshold로 대체하도록 동기를 부여하여 inference 중 decision errors를 줄일 수 있다. 

  설명의 편의를 위해 엔티티 쌍 의 라벨 T = (es, eo) 을 positive classes PT와 negative classes NT의 두 subsets으로 나누어 다음과 같이 정의하였다:

- positive classes PT ⊆ R은 T에 있는 엔티티 사이에 존재한다. 만약 T가 어느 relation에도 존재하지 않으면, PT는 empty이다.
- negatice classes NT ⊆ R은 엔티티 사이에 존재하지 않은 relation이다. 만약 T가 어느 relation에도 표현되지 않으면, NT = R 이다.

만약 한 엔티티 쌍이 맞게 분류된다면, positive class의 logit이 thresdhold보다 반드시 높아야하고, 반면에 negative class의 logit은 낮아야 한다. 여기서는 threshold class TH가 있다. 이 값은 다른 클래스와 동일한 방식으로 자동으로 학습된다. 테스트 시 TH 클래스보다 로짓이 높은 클래스를 positive classes로 반환하거나, 해당 클래스가 없는 경우 NA를 반환한다. 이 threshold class는 entities-dependent threshold value를 학습한다. 이것은 global threshold을 대체하므로 dev set에서 threshold를 조정할 필요가 없다. 

  새로운 모델을 배우기 위해서는 TH 클래스를 고려한 특별한 loss function이 필요하다. 우리는 standard categorical cross entropy loss을 기반으로 adaptive-thresholding loss을 설계한다.  loss function은 아래와 같이 두 부분으로 나뉜다.
$$
L_1 = - \sum_{r∈P_T}log \Bigg( \frac{exp(logit_r)}{\sum_{r'∈ P_T∪\{TH\}}exp(logit_{r'}) } \Bigg) , \\
L_2 = - log \Bigg( \frac{exp(logit_{TH})}{\sum_{r'∈N_T∪\{TH\}} exp(logit_{r'}) } \Bigg), \\
L = L_1 + L_2 \\
$$
처음 부분 L1은 positive classes와 TH class를 포함한다. 여러 개의 positive classes가 있을 수 있으므로 total loss은 모든 positive classes에서 categorical cross entropy losses의 합으로 계산된다. L1은 모든  positive classes의 로짓이 TH 클래스보다 높도록 푸시한다.  positive label이 없는 경우에는 사용되지 않는다. 

두번째 부분 L2는 negative classes와 threshold class를 포함한다. 이것은 TH 클래스가 True Label인 categorical cross entropy loss이다. 이것은 negative class의 로짓이 TH 클래스보다 낮게 당겨지도록 한다. 총 손실에 대해 두 부분을 간단히 합하면 된다.

![Screen Shot 2021-07-13 at 5.31.55 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-13 at 5.31.55 PM.png)

  제안된 adaptive-thresholding loss는 Figure 2에서 볼 수 있다. 이 기술은 실험 결과 global threshold에서 큰 성능 향상을 얻었다.

## Localized Context Pooling

  logsumexp pooling은 전체 문서에서 한 엔티티에 대한 모든 mentions의 임베딩을 누적하고 이 엔티티에 대해 하나의 embedding을 생성한다. 그런 다음 이 문서 수준 global pooling에 포함된 엔티티가 모든 엔티티 쌍의 classification에 사용된다. 그러나 엔티티 쌍의 경우 엔티티의 일부 컨텍스트가 관련이 없을 수 있다. 예를들어 Figure 1의 경우, John Stanistreet에 대한 두 번째 mention과 그 context는 엔티티 쌍(*John Stanistreet*, *Bendigo*)과 관련이 없다. 따라서 이 엔티티 쌍에 대한 관계를 결정하는 데 유용한 문서 내의 관련 컨텍스트에만 집중하는 localized representation을 사용하는 것이 좋다. 

  따라서 우리는 localized context pooling을 제안하며, 여기서는 두 엔티티와 관련된 추가적인 local context embedding으로 엔티티 쌍의 임베딩을 강화한다. 이 연구에서는 이미 multi-head self-attention을 통해 token-level dependencies를 학습한 인코더로 pre-trained transfer-based model을 사용하기 때문에, 해당 모델의 attention heads를 loacalized context pooling에 직접적으로 사용하는 것을 고려한다. 이 방법은 새로운 attention layers를 처음부터 배우지 않고 pre-trained language model에서 잘 학습된 dependencies를 전송한다.

  구체적으로, pre-trained multi-head attention matrix A ∈ R^{H×l×l} 가 주어졌을 때, A_ijk는 i번째 attention head에 있는 token j에서 token k의 attention을 나타내고, 우리는 먼저 mention-level attention으로써 "*" symbol에서 attention을 취하고, 동일한 엔티티의 mention에 대한 attention을 평균화하여 entity-level attention A^E_i ∈ R^{H×l}을 얻는다. 이것은 i번째 엔티티에서 모든 토큰에 대한 attantion을 나타낸다. 그런 다음 엔티티 쌍 (es,eo)가 주어졌을 때, 그들의 entity-level attention을 곱하여 es와 eo 모두에 중요한 local context를 찾고, localized context embedding c^(s,o)를 구한다.
$$
A^{(s,o)} = A^E_s · A^E_o, \\
q^{(s,o)} = \sum^H_{i=1} A_i^{(s,o)}, \\
a^{(s,o)} = q^{(s,o)}/ 1^\intercal q^{(s,o)} , \\
c^{(s,o)} = H^\intercal a^{(s,o)} \\

H : \mbox{ contextual embedding }
$$
Localized context embedding은 원래 linear layer Eq.3 , Eq.4를 다음과 같이 수정하여 서로 다른 엔티티 쌍에 대해 서로 다른 엔티티 representation을 얻기 위해 globally pooled entity embedding으로 융합된다.
$$
z_s^{(s,o)} = \tanh(W_sh_{e_s} + W_{c_1}c^{(s,o)}, \\
z_o^{(s,o)} = \tanh(W_oh_{e_o} + W_{c_2}c^{(s,o)} \\
W_{c_1}, W_{c_2} ∈ R^{d×d}  : \mbox{model parameters}
$$
제안된  localized context pooling은 Figure 3에 나와 있다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-13 at 6.34.40 PM.png" alt="Screen Shot 2021-07-13 at 6.34.40 PM" style="zoom:40%;" />

실험에서는 마지막 transformer layer의 attention matrix를 사용한다.

## Experiments

### Datasets

