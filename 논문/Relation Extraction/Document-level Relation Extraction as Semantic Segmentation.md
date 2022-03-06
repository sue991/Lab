# Document-level Relation Extraction as Semantic Segmentation

## Abstract

  문서 수준 관계 추출은 문서에서 여러 엔티티 쌍 사이의 관계를 추출하는 것을 목표로 한다. 이 논문은 컴퓨터 비전의 의미 분할 작업과 병행하여 로컬 및 글로벌 정보를 수집하기 위한 엔티티 수준 관계 매트릭스를 예측함으로써 문제에 접근한다. 여기서, 문서 레벨 관계 추출을 위한 Document U-shaped Network를 제안한다. 특히, 우리는 인코더 모듈을 활용하여 엔티티의 컨텍스트 정보를 캡처하고 이미지 스타일 기능 맵을 통해 U자형 분할 모듈을 캡처하여 트리플 간의 글로벌 상호 의존성을 캡처합니다. 실험 결과에 따르면 우리의 접근 방식은 DocRED, CDR 및 GDA의 세 가지 벤치마크 데이터셋에서 최첨단 성능을 얻을 수 있다.

## Introduction

 예전에는 single sentence에서 relation을 확인하는데에 초점을 맞췄지만, 실제론 multiple sentences에서 relations가 나타난다. 따라서 최근 연구는 sentence-level RE에서 document-level RE로 확장되었다. sentence에서 classify하기에 오직 하나의 entity pair가 포함되어있었던 sentence-level RE와 비교했을 때, doc-level RE는 동시에 multiple entity pairs의 relation을 classify하는 model이 필요하다. 게다가 relation에 포함되는 subject와 object entities가 다른 문장들에서 나타날 수 있다. 따라서 한 문장만으로 관계를 식별할 수 없다. 

  inter-sentence entity pairs의 relations을 추출하기 위해 대부분의 최신 연구는 휴리스틱, structed attention 또는 dependency structures에 기반한 doc-level graph module로 구조화되었고, graph neural models을 사용한 추론(resoning)을 실시했다. 한편 transformer 아키텍처를 사용하는 것은 long-distance dependencies를 모델링할 수 있다는 점을 고려하면서, 몇몇 연구는 explicit graph resoning보다는 pre-trained language model을 직접적으로 적용했다. 일반적으로 현재 접근방식은 doc-level graphs 또는 transformer-based structure learning의 노드를 통과하는 information을 통해 entity representation을 얻는다. 그러나, 이것은 entitiy pairs사이의 global interactions보다는 token-level syntactic features 또는 contextual information에 초점을 맞추고, 하나의 context에서 multiple relations 사이의 interdependency는 무시한다.

 구체적으로,  multiple triples의 interdependency는 장점이 많고, 많은 entities의 경우 relation classification에 대한 지침을 제공할 수 있다. 

 Multiple triples에서 interdependency를 캡처하기 위해, Figure 2에 나타낸 것처럼 문서 수준 RE 과제를 entity-level classification problem, table filling으로 재구성한다. ![Screen Shot 2021-07-02 at 1.49.13 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-02 at 1.49.13 PM.png)

이는 semantic segmentation(잘 알려진 컴퓨터 비전 작업)과 유사하다. 이 작업의 목표는 컨볼루션 네트워크로 표시되는 해당 클래스로 이미지의 각 픽셀에 레이블을 붙이는 것이다. 상기 내용에 영감을 받아, semantic segmentation로써 document-level RE를 구성하는 Document U-shaped Network (DocuNet)라는 새로운 모델을 제안한다. 이러한 방식으로 모델은 엔티티 쌍 간의 관련 기능을 image으로 지정하면 pixel-level mask로 각 엔티티 쌍의 relation type을 예측한다. 구체적으로, 우리는 엔티티의 context information을 캡쳐기 위해 encoder module을 도입하고, triple 간의 global inter-dependency를 캡쳐하기 위해 image-style feature map을 통해 U-shaped segmentation을 도입한다.(특히, 이미지 스타일 기능 맵을 통해 엔티티의 컨텍스트 정보와 U자형 분할 모듈을 캡처하여 트리플 간의 글로벌 상호 의존성을 캡처한다.) 또한 불균형 관계 분포(imbalance relation distribution)를 처리하기 위해 balanced softmax method를 제안한다. 우리의 기여는 다음과 같이 요약할 수 있다.

- 우리가 아는 한, 이것은 문서 수준 RE를 의미 semantic segmentation task로 간주하는 첫 번째 접근 방식이다.
- DocuNet 모델을 도입하여 문서 레벨 RE를 위한 triples 간의 local context information와 global interdependency을 모두 캡처한다.
- 3개의 벤치마크 데이터셋에 대한 실험 결과에 따르면 우리의 모델 DocuNet은 기준선에 비해 최첨단 성능을 달성할 수 있다.

## Related Work

  이전까지의 RE approaches는 주로 한 sentence에 있는 두 엔티티 사이의 relation을 identifying 하는 것에 집중했다. sentence-level RE 과제를 효과적으로 다루기 위해 많은 접근법이 제안되었다. 그러나, sentence-level RE는 많은 real-world relations가 여러 문장을 읽어야만 추출될 수 있다는 점에서 피할 수 없는 제한에 직면한다. 이러한 이유로, 문서 레벨 RE는 많은 연구자들에게 어필한다. 

  문서 레벨 RE에 대한 다양한 접근 방식에는 주로 graph-based models과 transformer-based models이 포함된다. Graph- based approaches는 relational reasoning의 효과와 강점으로 인해 RE에서 널리 채택되고 있다. Jia *et al.* [2019]는 문서 전체와 sub-relation hierarchy 전반에 걸쳐 학습된 표현을 결합하는 모델을 제안했다. Christopoulou *et al.* [2019]는 Doc-level RE를 위한 edge-oriented graph neural model (EoG)을 제안했다.  Li *et al.* [2020]는 graph-enhanced dual attention network (GEDA)를 통해 문장과 잠재적 관계 인스턴스 사이의 복잡한 상호작용을 특징지었다.  Zhang *et al.* [2020c] 는 structure modeling layer 다음에  relation reasoning layer이 포함된 새로운 그래프 기반 모델인 Dual-tier Heterogeneous Graph (DHG)를 제안했다. Zhou *et al.* [2020] 는 노드로서의 엔티티 및 노드 사이의 edges로서의 엔티티 쌍의 컨텍스트로 구성된 global context-enhanced graph convolutional network(GCN)를 제안했다. 그다음 GLRE, LSR, GAIN, HeterGSAN 등이 제시되었다. 명확한 그래프 추론(Explicit graph reasoning)은 서로 다른 문장에서 발생하는 entities 간의 차이를 메워 장거리 의존성을 완화하고 좋은 성과를 달성할 수 있다.

  대조적으로, 트랜스포머 아키텍처가 암시적으로 long-distance dependencies를 모델링 할 수 있다는 것을 고려할 때, 몇몇 연구자들은 doucument graphs를 생성하지 않고 pre-trained language models를 직접 활용한다. Wang *et al.* [2019] 은 사전 교육된 단어 임베딩으로 BERT를 사용하여 DocRED에 대한 two-step training paradigm을 제안했다. 그들은 관계 분포의 불균형을 관찰했고 더 나은 inference를 위해 relation identification 과 classification를 해체했다. Tang *et al.* [2020]은 BERT를 기반으로 한 적응형 임계값 지정 및 지역화된 컨텍스트 풀링의 새로운 변압기 기반 모델 (ATLOP)을 제안했다. 그러나, 대부분의 이전 연구들은 triples 간의 high-level global connections에 관계 없이, local entity representation에 초점을 맞추었고, 이는 multiple relations 사이의 interdependency을 간과하였다. 

  한편, 우리의 연구는 [Jin et al., 2020]에서 영감을 얻었는데, 이것은 관계 간의 글로벌 상호작용의 문제를 처음으로 고려한 것이었으며, RE에 관한 연구는 거의 없었다. 반면에, 이 연구들 [Nguyen and Grishman, 2015; Shen and Huang, 2016]과 같이, CNN는  relation extraction영역에서 오랫동안 사용되어 왔으며, 이를 통해 우리는 이미지 스타일 feature map의 정보 추출에 있어 CNN의 역할에 주의를 기울일 수 있다. 따라서, 우리의 연구는 또한 [Lu 등, 2020]의 연구와 관련있는데, 이 연구는 incomplete utterance rewriting를 semantic segmentation task로 공식화하고 컴퓨터 비전 관점에서 RE 문제를 연구하도록 동기를 부여했다. 본 연구에서는 contracting path로 구성된 U-Net [Ronneberger et al., 2015]을 활용하여 컨텍스트를 캡처하고 정확한 localization를 지원하는 symmetric expanding path를 활용했다. 우리가 아는 한, 이것은 RE를 semantic segmentation task로 공식화하기 위한 첫 번째 접근법이다.

## Methodology

### 1. Preliminary

  먼저 문제 정의를 소개한다. 엔티티 집합 {e_i}^n\_{i=1}을 포함하는 document *d* 의 경우, task는 entitiy pair (e_s, e_o) 사이의 relation을 추출하는 것이다. 하나의 문서에서 각 엔티티 e_i는 여러번 발생할 수 있다. e_s와 e_o 사이의 RE를 모델링하기 위해, N*N 매트릭스 Y를 정의하는데, 이때 엔티티 Y\_{s,o}는 e_s와 e_o사이의 relation type을 의미한다. 그리고 나서 우리는 semantic segmentation task와 유사한 matrix Y의 아웃풋을 획득한다. Y에 있는 엔티티는 문서의 첫번째 appearance에 따라 배열된다. 우리는 entity-to-entity relevance estimation을 통해 feature map을 얻고 feature map을 이미지로 가져온다. Output entity-level relation matrix Y는 semantic segmentation에서 pixel-level mask와 병렬로 되어 relation extraction과 semantic segmentation을 연결한다. 우리의 접근법은 sentence- level의 relation extraction에도 적용될 수 있다. 따라서 문서에 엔티티가 상대적으로 많으므로 엔티티 수준 관계 매트릭스에서 더 많은 global information를 학습하여 성능을 높일 수 있습니다.

### 2. Encoder Module

  Document d 가 주어졌을 때, mention의 처음과 끝 부분에 특별한 심볼 "<e>" 와  "</e>"를 주입하여 엔티티의 위치를 표시한다.
$$
d = [x_t]_{t=1}^L
$$
pre-trained language model을 인코더로 활용하여 다음과 같은 임베딩을 획득한다.
$$
H = [h_1,h_2,...,h_L] = Encoder([x_1,x_2,...,x_L]) \\
h_i  : \mbox{token } x_i \mbox{의 embedding}
$$
몇몇 doc는 512보다 길기 때문에 *dynamic window* 를 활용하여 전체 문서를 인코딩한다. 최종 representations을 얻기 위해 서로 다른 windows의 겹치는 토큰의 임베딩을 평균으로 한다. 그 후, 우리는 "<e>"의 임베딩을 이용하여 [Verga et al., 2018]에 따른 mention을 나타낸다. 우리는 원활한 버전의 max pooling, 즉 각 엔티티 ei를 얻기 위해 log-sumexp pooling[Jia 등, 2019]을 활용한다. 
$$
e_i = log\sum^{N_{e_i}}_{j=1}exp(m_j)
$$
이 풀링은 문서에 언급된 signals를 누적한다. 따라서 엔티티 임베딩 ei를 얻는다. 우리는 entity-to-entity relevance를 기반으로 entity-level relation matrix를 계산한다. 각각 매트릭스에 있는 entity ei에 대해, 그들의 relevance는 D-dimensional feature vector F(e_s,e_o)에 의해 캡쳐된다. F(e_s,e_o)를 계산하기 위한 두가지 전략, 즉 *similarity-based* method와 *context- based* method을 소개하겠다. Similarity-based method는 e_s와 e_o 사이의 element-wise similarity, cosine similarity 그리고 bi-linear similarity의 연산 결과를 concatenate함으로써 생성된다.
$$
F(e_s,e_o) = [e_s⊙e_o;cos(e_s,e_o);e_sW_1e_o]
$$
context-based strategy의 경우, affine transformation과 함께 entity-aware attention을 활용하여 feature vector를 얻는다.
$$
F(e_s,e_o) = W_2Ha^{(s,o)} \\
a^{(s,o)} = softmax(\sum^K_{i=1}A^s_i·A^o_i) \\
a^{(s,o)} : \mbox{entity-aware attention의 attention weight} \\
A^s_i : \mbox{i-th entity에 대한 토큰의 중요성} \\
H : \mbox{document embedding} \\
W_1, W_2 : \mbox{learnable weight matrix}
K : \mbox{transformer의 head 수}
$$


### 3. U-shaped Segmentation Module

entity-level relation matrix F ∈ R^{N×N×D}를 D-channel image로 취하면서, F에서  pixel-level mask로 document-level relation 예측을 공식화한다. 이때 N은 모든 데이터셋의 샘플에서 카운트 되는 가장 많은 수의 엔티티 이다. 이를 위해, 우리는 컴퓨터 비전의 유명한 semantic segmentation model인 U-Net [Ron-neberger et al., 2015]을 활용한다.

![Screen Shot 2021-07-11 at 4.34.08 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-11 at 4.34.08 PM.png)

Figure 3에서 볼 수 있는 것 처럼, 모듈은 skip-connection을 가진 두개의 down-sampling blocks와 두 개의 up-sampling blocks을 포함한 U-shaped segmentation structure로 형성된다. 한편, 각 다운-샘플링 블록은 두개의 후속 max-pooling과 별도 convolution 모듈이 있다. 또한 각 다운샘플링 블록에서 채널 수가 두 배로 증가한다. 그림 2에서 볼 수 있듯이, entity-level relation matrix의 segmentation area는 entity pairs 간의 관계가 동시에 발생하는 것을 가리킨다. U-shaped segmentation structure는 receptive field에서 implicit reasoning과 유사한 entity pairs 간의 정보 교환을 촉진할 수 있다. 구체적으로, CNN과 down-sampling block은 현재 엔티티 쌍 embedding F(e_s,e_o)의 receptive field를 확장할 수 있고, 따라서, representation learning을 위한 풍부한 global information를 제공할 수 있다. 반면에, 모델에는 후속 deconvolution neural network와 두 개의 별도 컨볼루션 모듈이 있는 두 개의 up-sampling blocks가 있다. 다운샘플링과 달리 각 업샘플링 블록에서 채널 수가 절반으로 줄어들어 집계된(aggregated) 정보가 각 픽셀로 분산될 수 있다. 마지막으로, 다음과 같이 로컬 및 글로벌 정보 Y를 모두 캡처하기 위해 인코딩 모듈과 U- shaped segmentation module을 통합한다.
$$
Y = U(W_3F) \\
U : \mbox{ U-shaped segmentation module} \\
Y ∈ R^{N×N×D} : \mbox{ entity-level relation matrix} \\
W_3 : \mbox{F의 dimension을 줄이기 위한 learnable weight matrix} \\
D' \mbox{은 D보다 훨씬 작다.}
$$

### 4. Classification Module

entity-level relation matrix Y와 함께 entity pair embedding **e_s**와 **e_o**가 주어졌을 때, feedforward neural network를 통해 hidden representations z에 매핑한다. 그 다음, 우리는 bilinear function을 통해 relation의 probability를 얻는다. 형식적으로 다음과 같이 얻는다.
$$
z_s = tanh(W_se_s + Y_{s,o}), \\
z_o = tanh(W_oe_o + Y_{s,o}) , \\
P(r|e_s,e_o) = \sigma(z_sW_rz_o + b_r), \\
Y_{s,o} : \mbox{matrix Y에 있는 (s,o)의 entity-pair representation} \\
W_r  ∈ R^{d*d} ; b_r  ∈ R ; W_s  ∈ R_{d*d} ; W_o  ∈ R^{d*d} : \mbox{learnable parameters}
$$
  이전 연구 [Wang *et al.*, 2019] 에서 RE에 대한 imbalance relation distribution(많은 entity pairs가 *NA* 의 relation을 갖고 있음.)을 관찰했기 때문에, 우리는 balanced softmax 방법을 training에 도입했고, 이것은 Computer Vision에서 circle loss [Sun *et al.*, 2020] 에서 영감을 얻었다. 구체적으로 우리는 target category의 scores가 모두 s0보다 크고 non-target categories의 scores는 모두 s0보다 작기를 바라는 추가적인 카테고리 0을 도입했다. 식은 다음과 같이 나타난다.
$$
L = log\Bigg( e^{s_0} + \sum_{i∈Ω_{neg}}e^{s_i} \Bigg) + log \Bigg( e^{-s_0} + \sum _{j∈Ω_{pos}} e^{-s_j} \Bigg)
$$
더 간단하게, threshold를 zero로 설정하고 다음과 같은 식을 얻는다.
$$
L = log \Bigg( 1 + \sum_{i∈Ωneg}e^{s_i} \bigg) + log \Bigg( 1 + \sum_{j∈Ωpos} e^{-s_j} \Bigg)
$$

## Experiments

### 1. Dataset

  우리는  DocuNet model을 세 개의 document-level RE datasets으로 평가했다. Table 1에 데이터셋 통계를 나열했다

- **DocRED** [Yao *et al.*, 2019]은 크라우드소싱에 의한 대규모 문서 레벨 관계 추출 데이터 세트이다. DocRed는training, validating, test 각각 3,053/1,000/1,000 개의  instances 를 갖고 있다.
- **CDR** [Li *et al.*, 2016] 은 biomedical 영역의 relation extraction dataset이고, chemical과 disease concepts 사이의 interaction 추론에 중점을 두고 있다.
- **GDA** [Wu *et al.*, 2019] 은 23,353 training samples로 구성된 biomedical 영역 데이터셋이다. 다른점으로, 이 데이터셋은 disease concepts와 genes 사이의 interaction 예측에 중점을 두고 있다.

### 2. Experimental Settings

  우리 모델은 Pytorch에서 수행된다. 우리는 DocRED의 인코더로써 cased BERT-base 또는 RoBERTa-large를, CDR and GDA의 SciBERT-base를 사용했다. 우리는 AdamW를 사용하여 learning rate 2e-5와 처음 6% steps의 linear warmup을  통해 모델을 optimize한다. Matrix size N = 42로 설정한다. Context-based strategy는 default로 활용된다. 우리는 development set으로 hyperparameters를 조율한다. 우리는 하나의 NVIDIA V100 16GB GPU로 트레이닝하고 [Yao et al., 2019]에 이어 Ign F1, F1로 모델을 평가한다.

![Screen Shot 2021-07-12 at 12.29.00 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-12 at 12.29.00 PM.png)

### 3. Results on DocRED Dataset

  우리는 DocuNet을 DocRED Dataset에 있는 GEDA, LSR, GLRE, GAIN, HeterGSAN을 포함한 graph-based models과 BERT_base, BERT-TS_base, HIN-BERT_base, CorefBERT_base, ATLOP_base를 포함한 tranformer-based model과 비교했다. Table 2에서, DocuNet-BERT_base가 ATLOP-BERTbase보다 더 좋은 성능을 낸 것을 볼 수 있고, DocuNet 모델이 새로운 RoBERTa-large와 함께 state-of-the-art result를 낸 것을 볼 수 있다.  2021년 1월 20일 IJCAI 마감일을 기준으로 당사는 외부 데이터 없이 DocutNet이라는 별칭으로 CodaLab 점수판2에서 1위를 차지했다.

### 4.Results on the Binomedical Datasets

Biomedical 데이터셋에서, DocuNet을 BRAN,EoG,LSR,DHG,GLRE, ATLOP의 기준선과 비교한다. ATLOP 에 이어, 우리는 SciBERT 를 활용하는데, 이것은 scientific publication corpora에 대해 pre-trained 되었다. Table 3에서, 우리는 DocumentNet-SciBERT 베이스가 ATLOP-SciBERT 베이스와 비교하여 CDR과 GDA에서 F1 점수를 6.9%, 1.4% 향상시킨 것을 관찰했다.

### 5. Alation Study

  우리는 접근법의 다양한 구성요소의 효과를 검증하기 위해 ablation study experiment을 수행했다. **DocNet(Similarity-based)**은 context-based strategy이 아니라 두 엔티티 간의 상관 관계를 입력 매트릭스로 계산하기 위해 similarity functions strategy을 직접 사용하는 것을 의미한다. **w/o U-shaped Segmentation**은 segmentation module이 feed-forward neural network로 대체됨을 의미한다. Table 4를 보면, 각 모듈이 없는 모든 모델의 성능이 저하되는 것을 알 수 있으며, 이는 두 components가 모두 유익하다는 것을 나타낸다.

![Screen Shot 2021-07-12 at 1.15.10 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-12 at 1.15.10 PM.png)

  또한, U-shaped segmentation module과 balanced softmax module은 성능을 모델링하는 데 가장 중요하며 F1에 민감하여 DocNet에서 제거하면 개발 F1 점수가 2.18%와 1.32% 하락하는 것으로 확인되었다. 이는 우리의 모델이 포착한 triples간의 global interdependency가 이 문서 레벨 RE에 효과적이라는 것을 보여줍니다. 또한,  context-based strategy과 비교하여 similarity functions strategy에 기반한 접근 방식은 0.84 F1만큼 떨어지며, 이는 context-based strategy이 유리함을 보여준다.

### 6. Case Study

  우리는 GAIN [Zen 등, 2020]에 따라 동일한 사례를 선택하고 기준선과 비교한 우리의 모델 DocuNet의 효과를 더욱 자세히 설명하기 위한 사례 연구를 수행한다. Figure 4와 같이 BERT_base와 DocuNet-BERT_base 모두 "Without Me"와 "The Eminem Show" 사이의 "part of" relation을 성공적으로 추출할 수 있다.

![Screen Shot 2021-07-12 at 1.29.27 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-12 at 1.29.27 PM.png)

그러나 우리 모델 DocuNet-BERT 베이스만 "Without Me"의 "performer"와 "publication date"가 각각 "Eminem"과 "May 26, 2002"와 같다는 것을 추론할 수 있다. 우리는 직관적으로 이러한 엔티티 사이에서 위에서 언급한 관계 추출이 문장 전체에 걸쳐 logical inference을 필요로 한다는 것을 관찰할 수 있다. 이러한 흥미로운 관찰은 entity-level 관계 매트릭스에 대한 U-shaped segmentation structure가 엔티티 간에 relational reasoning을 암시적으로 수행할 수 있음을 나타낸다.

![Screen Shot 2021-07-12 at 1.39.16 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-12 at 1.39.16 PM.png)

### 7. Analysis

  다중 엔터티의 글로벌 정보 모델링에 있어 DocuNet의 효과를 평가하기 위해 DocRED에 설정된 여러 development set에 대해 U-shaped segmentation module을 포함하거나 포함하지 않은 모델을 각각 평가하였다. Figure 5에서, U자형 분할 모듈이 있는 모델이 U자형 분할 모듈이 없는 모델을 일관되게 능가하는 것을 관찰한다. 우리는 엔티티 수가 증가하면 더 많은 향상이 일어난 다는 것에 주목했다. 이는 U자형 분할 모듈이 하나의 컨텍스트에서 여러 triples 간의 상호의존성을 암묵적으로 학습할 수 있다는 것을 의미하며, 따라서 문서 수준의 RE 성능을 향상시킨다. 

## Conclusion and Future Work

  본 연구에서는 문서 수준 RE를 의미 분할 과제로 공식화하고 Document U자형 네트워크를 도입하는 첫 단계를 밟았다. 실험 결과, 우리 모델은 기준선보다 로컬 및 글로벌 정보를 캡처하여 더 나은 성능을 달성할 수 있는 것으로 나타났다. 우리는 또한 엔티티-엔터티 관계 매트릭스에 대한 컨볼루션이 엔티티 간에 관계적 추론을 암시적으로 수행할 수 있다는 것을 경험적으로 관찰했다. 향후, 우리는 측면 기반 정서 분석 및 네스트 네임 인식과 같은 다른 범위 차원의 분류 작업에 우리의 접근 방식을 적용할 계획이다.

