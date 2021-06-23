# DocRED: A Large-Scale Document-Level Relation Extraction Dataset

## Abstract

 문서 내 multi entities는 일반적으로 inter-sentence relations를 나타내며, single entity pairs를 위해 일반적으로 intra-sentence relations에 초점을 맞춘 existing relation extraction(RE) 방법에 의해 잘 처리될 수 없다. document-level RE에 대한 연구를 가속화하기 위해, 우리는 위키피디아와 위키데이터에서 생성된 새로운 데이터 세트인 **DocRED**를 소개합니다. 이 DocRED는 세가지 특징이 있는데, 

1. DocRED는 named entities와 relations 모두를 annotate하며, plain text에서 document-level RE를 위한 가장 큰 human-annotated dataset이다.

2. DocRED는 여러 문장들을 문서에서 판독하여 entity를 추출하고 문서의 모든 정보를 합성함으로써 그들의 relation를 추론하도록 요구한다.
3. human-annotated data와 마찬가지로, DocRED가 supervised 그리고 weakly supervised scenarios에 채택될 수 있도록 하는 large-scale distantly supervised data를 제공한다.

Document-level RE의 과제를 검증하기 위해 RE를 위한 최신 연구를 구현하고 DocRED에서 이러한 방법의 철저한 평가를 수행한다. 경험적 결과는 DocRED가 기존의 RE 방법들에 도전하고 있음을 보여 주고 있으며, 이는 document-level  RE가 개방된 문제로 남아 있고 추가 노력이 필요하다는 것을 보여준다. 실험들에 대한 상세한 분석을 기반으로, 우리는 향후 연구를 위한 여러 가지 약속 방향들을 논의한다. 우리는 DocRED와 baselines 코드를 다음 [주소](https: //github.com/thunlp/DocRED)에서 공개적으로 사용할 수 있게 한다. 

## 1. Introduction

relation extraction(RE)의 과제는 large-scale knowledge graph construction에 중요한 역할을 하는 일반 텍스트에서 entities 간의 relational facts을 식별하는 것이다. 대부분의 기존 RE 작업은 sentence-level RE, 즉 single sentence에서 관계 사실을 추출하는 데 중점을 둔다. 최근 몇 년 동안 sentence-level RE에 대한 엔티티의 relational patterns을 인코딩하고 최첨단 성능을 달성하기 위해 다양한 뉴럴 모델이 탐색되었다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-20 at 5.33.34 PM.png" alt="Screen Shot 2021-06-20 at 5.33.34 PM" style="zoom:50%;" />

 이런 성공적인 노력에도 불구하고, sentence-level RE는 실무에서 피할 수 없는 제약을 겪는데, 그것은 많은 relational facts가 multiple sentences에 나타나는 것이다. Figure 1을 예로 들면, 문서에는 multiple entities가 언급되어 있으며 복잡한 interactions를 나타낸다. relational fact(*Riddarhuset*, country, *Sweden*)를 확인하기 위해서, 먼저 문장 4에서 *Riddarhuset*이 *Stockholm*에 위치해 있다는 사실을 확인한 후, 문장 1로부터 *Stockholm*이 *Sweden*의 수도이고 *Sweden*이  country라는 사실을 확인하고, 마지막으로 이 facts로부터 Riddarhuset의 주권국가(sovereign state)가 *Sweden*이라는 사실을 추론해야 한다.  이 프로세스는 문서의  multiple sentences에 대한 읽기 및 추론을 필요로 하며, 이는 직관적으로 sentence-level RE methods의 범위를 벗어난다. 위키피디아 문서로부터 샘플링된 human-annotated corpus에 대한 통계에 따르면, 최소 40.7%의 relational facts가 multiple sentences로부터 추출될 수 있으며, 이는 무시할 수 없다. Swampillai and Stevenson (2010)과 Verga et al. (2018)도 유사한 관측 결과를 보고했다. 따라서 RE를 sentence level에서 document level으로 이동할 필요가 있다. 

 Document-level RE에서의 연구는 training과 evaluation에서 large-scale annotated dataset을 필요로 했다. 현재 document-level RE에서 dataset은 많지 않다.  Quirk and Poon (2017)과 Peng et al. (2017)은 human annotation 없이 두 개의 distantly supervised datasets을 구축하여 평가의 신뢰성이 떨어질 수 있다. BC5CDR(Li et al., 2016)은 1,500개의 PubMed 문서로 구성된 human-annotated document-level RE dataset이며, 이는 "화학 유발 질병" 관계만을 고려한 생약의 특정 영역에 있으므로 문서 수준 RE에 대한 범용 방법을 개발하는 데 적합하지 않다. Levy et al. (2017)은  reading comprehension 방법을 사용하여 질문에 답함으로써 문서에서 relational facts을 추출하며, 여기서 질문은 entity-relation 쌍에서 변환된다. 본 연구에서 제안된 데이터 세트는 특정 접근 방식에 따라 맞춤화되므로, document-level RE에 대한 다른 잠재적 접근 방식에도 적합하지 않다. 요약하면, document-level RE에 대한 기존 데이터 세트는 manually-annotated relations와 entities의 수가 적거나, distant supervision에서 noisy annotations을 보여주거나, 특정 도메인 또는 접근 방식을 제공한다. document-lebel RE에 대한 연구를 가속화하기 위해, large-scale, manually-annotated, 그리고 general-purpose document-level RE dataset이 시급하다.

 이 논문에서, 우리는 위키백과와 위키데이터로 구성된 large-scale human-annotated document-level RE dataset인 DocRED를 제시한다. DocRED는 다음 세 가지 특징으로 구성된다.

1. DocRED는 5,053개의 위키백과 문서에 annotated된 132,375개의 entities와 56,354개의 relational facts을 포함하고 있어 human-annotated document-level RE dataset 중 가장 크다.
2. DocRED에서 relational facts의 최소 40.7%는 multiple sentences에서 추출할 수 있으므로, DocRED는 entities를 인식하기 위해 문서의 여러 문장을 읽고 문서의 모든 정보를 합성하여 이들의 관계를 추론해야 한다. 이는 DocRED를 sentence-level RE datasets와 구별한다.
3. 또한 우리는 weakly supervised RE research를 위해 large-scale distantly supervised data를 제공한다.

  DocRED의 과제를 평가하기 위해 최신 RE 방법을 구현하고 다양한 설정에서 DocRED에 대한 철저한 실험을 수행한다.

실험 결과는 DocRED에서 기존 방법의 성능이 크게 저하된다는 것을 보여주며, document-level RE가 sentence-level RE보다 더 어렵고 여전히 열린 문제로 남아 있음을 보여준다. 또한, 결과에 대한 자세한 분석은 추구할 가치가 있는 여러 유망한 방향을 보여준다.

## 2. Data Collection

우리의 최종 목표는 일반 텍스트에서 document-level RE에 대한 dataset을 구성하는 것인데, 여기에는 named entity mentions, entity coreferences 및 문서의 모든 entity 쌍의 relations를 포함하는 필수 정보가 필요하다. 더 많은 RE 설정을 용이하게 하기 위해 relation instances에 대한 근거 정보도 제공한다. 다음 섹션에서는 먼저 human-annotated data의 collection 프로세스를 소개한 다음, large-scale distantly supervised data를 생성하는 프로세스를 설명한다.

### 2.1 Human-Annotated Data Collection

human-annotated data는 4 단계로 수집되었다.

1. 위키피디아 문서에 대해 distantly supervised annotation을 생성한다.
2. 문서(document) 및 참조(coreference) information에 있는 모든 named entity mentions에 Annotating한다.
3. named entity mentions을 Wikidata items에 연결한다.
4. relations 및 이에 상응하는 supporting evidence를 라벨링한다.

  ACE annotation process(Doddington et al., 2004)에 따르면 2단계와 4단계 모두 데이터에 대한 세 번의 반복 패스가 필요하다.

1. Named Entity Recognition (NER) models을 사용하여 named entity를 생성하거나, distant supervision 및 RE models를 사용하여 relation recommendations를 생성한다.
2. recommendations을 수동으로 수정 및 보완한다.
3. 더 나은 정확성과 일관성을 위해 두 번째 단계의 annotation results를 검토하고 추가로 수정한다. annotators가 잘 training되었는지 확인하기 위해 원칙적인 training 절차를 채택하고 annotators가 데이터 세트에 annotating하기 전에 테스트 작업을 통과해야 한다. 세 번째 pass annotation에는 신중하게 선택된 experienced annotators만 사용할 수 있다.

  텍스트와 KBs 간에 강력한 alignment를 제공하기 위해, 우리의 데이터 세트는 완전한 English Wikipedia document collection과 위키피디아와 긴밀하게 통합된 large-scale KB인 위키데이터로 구성된다. 우리는 위키백과 문서의 introductory 섹션을 corpus로 사용하는데, 그것들은 대개 고품질이며 대부분의 핵심 정보를 포함하고 있기 때문이다.

### Stage 1: Distantly Supervised Annotation Generation

Human annotation을 위한 documents를 선택하기 위해, 우리는 distant supervision assumption하에 위키백과 문서를 위키데이터와 정렬한다(Mintz et al., 2009). 구체적으로, 우리는 먼저 spaCy를 사용하여 named entity recognition을 수행한다. 그런 다음 이러한 named entity mentions은 동일한 KB ID를 가진 named entity mentions이 병합되는 Wikidata items와 연결됩니다. 마지막으로, 문서에서 병합된 각 named entity pair간의 relations는 Wikidata를 쿼리하여 라벨링된다. 128개 이하의 단어를 가진 documents는 무시된다.  추론을 장려하기 위해, 우리는 4개 미만의 entities 또는 4개 미만의 relation instances를 포함하는 문서를 추가로 폐기하여 결과적으로 distantly supervised labels가 있는 107,050개의 문서가 생성되며, 여기서 무작위로 5,053개의 documents와 가장 빈번한 human annotation를 위한 96개 relations를 선택한다.

### Stage 2: Named Entity and Coreference Annotation

 Document로부터 relations를 추출하려면 먼저 named entity mentions를 인식하고 doc 내에서 동일한 entities를 참고하는 mentions을 identifying 해야한다. High-quality named entity mentions 및 coreference information을 제공하기 위해, 우리는 먼저 human annotators에게 Stage 1에서 생성된  named entity mention recommendations을 검토하고 수정 및 보완한 다음, 같은 entities를 참조하는 서로 다른 mentions를 merge하도록 하며, 이는 추가적인 coreference information을 제공한다.  intermediate corpus 결과는 앞에서 언급한 유형에 속하지 않는 기타 엔티티의 사람, 위치, 조직, 시간, 숫자 및 이름을 포함하여 다양한 명명된 엔티티 유형을 포함한다.

### Stage 3: Entity Linking

 이 단계에서, 우리는 각각 named entity mention을 여러 Wikidata items에 연결하여 다음 Stage에 대한 distant supervision에서 relation recommendations를 제공한다. 구체적으로, 각 named entity mention은  이름 또는 별칭이 문자 그대로 일치하는 모든 Wikidata items로 구성된 Wikidata item candidate set과 연결된다. 우리는 또한 document authors에 의해  named entity mention과 hyperlinked된 Wikidata items와 entity linking toolkit TagMe(Ferragina and Scaiela, 2010)로 만든 recommendations를 사용하여 candidate set를 확장한다. 특히 숫자와 시간은 의미적으로 일치한다.

### Stage 4: Relation and Supporting Evidence Collection

 relation 및 supporting evidence의 annotation은 Stage 2에 있는 named entity mentions와 coreference information에 기초하여, 두 가지 주요 과제에 직면해 있다. 첫 번째 과제는 문서에 있는 많은 수의 potential entity pairs에서 비롯된다. 한편, 문서에서 엔티티 수(평균 19.5개의 엔티티)와 관련된 잠재적 엔티티 쌍의 2차 수를 고려할 때, 각 엔티티 쌍 간의 관계를 남김없이(전부) 라벨링하면 집약적인 workload(작업량)가 발생할 수 있다. 반면, 문서에 있는 대부분의 entity pairs은 relations를 포함하지 않는다. 두 번째 과제는 데이터 세트에서 세분화된 많은 relation types에 있다. 따라서 annotators가 처음부터 relation을 레이블링하는 것은 가능하지 않다.

 우리는 human annotators에게 RE 모델의 recommendations과 엔티티 링크(Stage 3)에 기초한 distant supervision을 제공함으로써 문제를 해결한다. 평균적으로, 우리는 entity linking에서 document 당 19.9개의 relation instance를 권장하고, 보완을 위해 RE 모델에서 7.8개를 권장한다. annotators에게 recommendations를 검토하고 잘못된 relation instances를 제거하고 누락된 항목을 보완하도록 요청한다. 보존된 relations는 외부 세계 지식에 의존하지 않고 문서에 반영되어야 한다. 마지막으로 entity linking으로부터 relation instances 57.2%, RE 모델의 relation instances 48.2%가 보존(reserved)된다.

### 2.2 Distantly Supervised Data Collection 

Human-annotated data 외에도, 우리는 weakly supervised RE scenarios를 촉진하기 위해 large-scale distantly supervised data를 수집한다. 우리는 106,926개의 documents에서 5,053개의 human-annotated documents를 제거하고, 나머지 101,873개의 documents를  distantly supervised data의 corpus로 사용한다. Distantly supervised data와 human-annotated data가 동일한 entity distribution을 공유하도록 하기 위해, named entity mentions은 2.1 Section에서 수집된 human-annotated data에 미세 조정되고 90.5% F1 score를 달성하는 Transformers의 Bidirectional Encoder Representations(BERT) (Devlin et al., 2019)을 사용하여 다시 식별된다. 우리는 target Wikidata item의 빈도 및 현재 문서와의 관련성을 공동으로 고려하는 heuristic-based method으로 각 named entity mention를 하나의 Wikidata item에 연결한다. 그런 다음 명named entity mentions을 동일한 KB IDs와 병합합니다. 마지막으로, 각 병합된 엔티티 쌍 간의 relations는 distant supervision을 통해 레이블링된다.

## 3. Data Analysis

 이 섹션에서는 DocRED의 다양한 측면을 분석하여 document-level RE의 데이터 세트와 task에 대한 더 깊은 이해를 제공한다.

![Screen Shot 2021-06-21 at 3.37.58 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-21 at 3.37.58 PM.png)

**Data Size.**  Table 1은 DocRED 및 sentence-level RE 데이터 세트 SemEval-2010 Task 8(Hendrickx 등, 2010), ACE 2003-2004(Doddington 등, 2004), TACRED(장 등, 2017), FewRel(Han 등, 2018B) 및 document- level RE dataset BC5CDR (Li et al., 2016)을 포함한 일부 대표적인 RE 데이터 세트의 통계를 보여준다. 우리는 DocRED가 문서, 단어, 문장, 엔티티, 특히  relation type, relation instances 및 relational facts 측면을 포함하여 많은 측면에서 기존 데이터 세트보다 크다는 것을 발견했다. large-scale DocRED 데이터 세트가 문장 수준에서 문서 수준으로 관계 추출을 추진할 수 있기를 희망한다.

**Named Entity Types.**  DocRED는 사람(18.5%), 위치(30.9%), 조직(14.4%), 시간(15.8%), 숫자(5.1%) 등 다양한 entity types를 포함한다. 또한 이벤트, 예술 작품, 법률 등 앞에서 언급한 유형에 속하지 않는 다양한 기타 entity names(15.2%)을 포함한다. 각 entity는 평균적으로 1.34개의 언급으로 주석을 달았다.

**Relation Types.**  우리의 데이터 세트는 Wikidata의 96개의 빈번한 relation types를 포함한다. 우리의 데이터 세트의 주목할 만한 특성은 relation types가 과학(33.3%), 예술(11.5%), 시간(8.3%), 개인 생활(4.2%) 등과 관련된 관계를 포함한 광범위한 범주를 포함한다는 것인데, 이는 relation facts가 특정 도메인에서 제약되지 않는다는 것을 의미한다. 또한, relation types은 문서 수준 RE 시스템에 풍부한 정보를 제공할 수 있는 잘 정의된 계층 및 분류법으로 구성됩니다.

**![Screen Shot 2021-06-21 at 4.22.35 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-21 at 4.22.35 PM.png)**

**Reasoning Types.** 3,820개의 관계 인스턴스를 포함하는 dev 및 test 세트에서 무작위로 300개의 문서를 샘플링하고 이러한 관계를 추출하는 데 필요한  reasoning types을 수동으로 분석하였다. Table 2는 데이터 세트의 주요 reasoning types에 대한 통계를 보여준다. reasoning types에 대한 통계에서 우리는 다음과 같은 관측치를 가지고 있다.

1. 대부분의 relation instances(61.1%)는 추론을 식별해야 하며, 간단한 패턴 인식을 통해 38.9%의 관계 인스턴스만 추출할 수 있어 문서 수준 RE에 reasoning이 필수적이라는 것을 알 수 있다.
2. 추론과 관련된 경우, majority(26.6%)는 논리적 추론을 필요로 하는데, 여기서 문제의 두 entities 사이의 관계는 bridge entity에 의해 간접적으로 설정된다. Logical reasoning은 RE 시스템이 여러 엔티티 간의 상호작용을 모델링할 수 있어야 한다.
3. 유의한 수의 relation instances(17.6%)는 coreference reasoning을 필요로 하는데, 여기서 coreference resolution은 풍부한 맥락(rich context)에서 대상 엔터티를 식별하기 위해 먼저 수행되어야 한다.

4. 유사한 비율의 elation instances(16.6%)는 상식적 추론(common-sense reasoning)을 기반으로 식별되어야 하며, 여기서 독자는 relation identification을 완료하기 위해 문서의 relational facts과 common-sense(상식)을 결합해야 한다.

**Inter-Sentence Relation Instances.**  각 relation instance는 평균 1.6개의 supporting sentences과 연관되어 있으며, 46.4%의 relation instances가 하나 이상의 supporting sentences과 연결되어 있음을 발견했다. 게다가 상세 분석 결과, 40.7%의 relational facts은 여러 문장으로부만 추출할 수 있어 DocRED가 document-level RE에 대한 좋은 벤치마크임을 알 수 있다. 우리는 또한 multiple sentence에 대한 읽기, 합성 및 추론 능력이 document-level RE에 필수적이라는 결론을 내릴 수 있다.

![Screen Shot 2021-06-21 at 4.34.00 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-21 at 4.34.00 PM.png)

## 4. Benchmark Settings

우리는 각각 supervised와 weakly supervised scenarios에 대한 두 가지 벤치마크 설정을 고안한다. 두 가지 설정 모두에 대해, RE 시스템은 고품질 human-annotated dataset에서 평가되며, 이는 문서 수준 RE 시스템에 대한 보다 신뢰할 수 있는 평가 결과를 제공한다. 두 설정에 사용된 데이터 통계는 Table 3에 나와 있다.

**Supervised Setting.**  이 설정에서는 무작위로 training, dev 및 test sets로 분할된 human- annotated data만 사용된다. Supervised setting은 문서 레벨 RE 시스템에 다음과 같은 두 가지 과제를 제기한다.

 첫 번째 과제는 문서 수준 RE 수행에 필요한 풍부한 추론 기술(rich reasoning skills)에서 비롯된다. Sec. 3 에서 보듯이, 약 61.1%의 relation instances는 추출해야 하는 패턴 인식 이외의 복잡한 추론 기술(reasoning skills)에 의존하며, 이를 위해서는 RE 시스템이 한 문장에서 단순한 패턴을 인식하는 것을 넘어서는 단계를 거쳐야 하고, 문서의 전역적이고 복잡한 정보에 대한 추론도 필요하다.

 두 번째 과제는 긴 문서를 모델링하는 높은 계산 비용(computational cost)과 문서의 잠재적 엔티티 쌍의 방대한 양에 있다. 이는 문서의 엔티티 번호(평균 19.5개 엔티티)와 관련하여 quadratic이다. 

결과적으로 (Sorokin 및 Gurevych, 2017; Christoplou et al., 2018)와 같은 2차 이상의 계산 복잡성의 알고리듬으로 컨텍스트 정보를 모델링하는 RE 시스템은 문서 수준 RE에 충분히 효율적이지 않다. 따라서 문서 수준 RE에 적용할 수 있도록 context-aware RE systems의 효율성을 더욱 개선할 필요가 있다.

**Weakly Supervised Setting.**  이 설정은 training set이 distantly supervised data로 대체된다는 점을 제외하고 supervised setting과 동일하다(Sec. 2.2). 앞에서 언급한 두 가지 과제 외에도, distantly suvpervised data와 동반되는 불가피한 잘못된 라벨링 문제는 weakly supervised setting에서 RE 모델의 주요 과제이다. Sentence-level RE에서 잘못된 라벨링 문제를 완화하기 위해 많은 노력이 들여져 왔다(Ride et al., 2010; Hoffmann et al., 2011; Surdeanu et al., 2012; Lin et al., 2016). 그러나 document-level의 distantly supervised data의 노이즈는 문장 수준의 노이즈보다 훨씬 더 크다.  예를 들어, human-annotated data collection의 Stage 4에서 head와 tail entities가 같은 문장(즉, 문장 내 관계 인스턴스)에서 동시에 발생하는 recommended relation instances의 경우, 41.4%는 부정확한 것으로 분류되는 반면, 61.8%의 문장 간(inter-sentence) relation instances는 부정확한 것으로 분류되어 잘못된 레이블링 문제를 나타낸다.  weakly supervised document-level RE의 경우 더 어렵다. 따라서, 우리는 DocRED에서 distantly supervised data를 제공하는 것이 문서 수준 RE에 대한distantly supervised methods의 개발을 가속화할 것이라고 믿는다. 또한, RE 시스템의 성능을 더욱 향상시키기 위해 distantly supervised data와 human-annotated data를 공동으로 활용할 수도 있다.

## 5. Experiments

 DocRED의 과제를 평가하기 위해 데이터 세트에서 최첨단 RE 시스템을 평가하기 위한 포괄적인 실험을 수행한다. 구체적으로, 우리는 supervised와 weakly supervised benchmark settings 모두에서 실험을 수행한다. 또한 우리는 human performance를 평가하고 다양한 supporting evidence types에 대한 성능을 분석한다. 또한 다양한 기능의 기여도를 조사하기 위해 ablation study(???)를 수행한다. 자세한 분석을 통해 문서 수준 RE에 대한 몇 가지 향후 방향에 대해 논의한다. 자세한 분석을 통해 문서 수준 RE에 대한 몇 가지 향후 방향에 대해 논의한다.

**Models.**  우리는 4개의 최신 RE model을 document-level RE scenario에 적용시키는데, 각각 CNN (Zeng et al., 2014) 기반 모델, LSTM (Hochreiter and Schmidhuber, 1997)기반 모델, bidirectional LSTM (BiLSTM) (Cai et al., 2016)기반 모델, 그리고 원래  contextual relation를 활용하여 intra-sentence RE를 개선하도록 설계된 Context-Aware 모델(Sorokin and Gurevych, 2017) 이 있다. 처음 세 가지 모델은 문서 인코딩에 사용되는 인코더에서만 다르며 이 섹션의 나머지 부분에서 자세히 설명될 것이다. 공간 제한에 대한 Context-Aware model의 세부 정보는 original paper를 보도록 한다.

 CNN/LSTM/BiLSTM 기반 모델은 먼저 CNN/LSTM/BiLSTM를 인코더로 사용하면서, n개의 어로 구성된 document D를 hidden state vector sequence h_i로 인코딩한 다음, 
$$
D = \{w_i\}^n_{i=1} \\
\{h_i\}^n_{i=1} \mbox{  : hidden state vector sequence }
$$
엔티티에 대한 표현을 계산하고 마지막으로 각 엔티티 쌍에 대한 관계를 예측한다.

 각 단어에 대해, 인코더에 제공되는 features은 GloVe word embedding(Penington et al., 2014), entity type embedding 및 coreference embedding의 concatenation이다. entity type embedding은 임베딩 매트릭스를 사용하여 단어에 할당된 엔티티 타입(예: PER, LOC, ORG)을 벡터에 매핑함으로써 얻어진다. entity type은 human-annotated data에 대해 인간에 의해 할당되고 distantly supervised data에 대해 미세 조정된 BERT 모델에 의해 할당된다. 동일한 엔티티에 해당하는 Named entity mentions는 문서에 처음 나타나는 순서에 따라 결정되는 동일한 엔티티 ID로 할당된다. 그리고 엔티티 ID는 coreference embeddings으로 벡터에 매핑된다.

 s번째 단어부터 t번째 단어까지 범위인 각 named entity mention *m_k*에 대해 그 표현을 다음과같이 정의한다.
$$
m_k = \frac{1}{t-s+1}\sum^t_{j=s}h_j
$$
그리고 K개의 mentions인 entity e_i의 표현은 이러한 멘션의 평균으로 계산된다.
$$
e_i = \frac{1}{K}\sum_km_k
$$
 우리는 relation prediction을 multi-label classification problem로 취급한다. 특히, 각 entity 쌍 (e_i, e_j)에 대해 먼저 entity representations을 relative distance embeddings과 concatenate한 다음 bilinear function를 사용하여 각 relation type에 대한 확률을 계산한다.
$$
\hat e_i = [e_i;E(d_{ij})], \hat e_j = [e_j;E(d_{ji})] \\
P(r|e_i,e_j) = sigmoid(\hat e^T_iW_r\hat e_j + b_r) \\
$$

$$
[·; ·] : \mbox{concatenation} \\ 
 d_{ij}, d_{ji} : \mbox{문서에 포함된 두 entity에 대한 첫 번째 mentions의 relative distances} \\
E : \mbox{embedding matrix  ;  }  R : \mbox{relation type} \\
W_r , b_r : \mbox{ relation type에 따른 trainable parameters}
$$

**Evaluation Metrics.**  널리 사용되는 두 가지 metrics F1과 AUC가 우리 실험에 사용된다. 그러나 training과 dev/test sets 모두에 존재하는 몇몇 relational facts 때문에 모델은 training 중에 이들의 relations를 암기하고 바람직하지 않은 방식으로 dev/test sets에서 더 나은 성능을 달성하여 evaluation bias를 도입할 수 있다. 그러나 traning과 dev/test sets의 간의 relational facts의 중복은 불가피하다. 많은 공통 relational facts가 서로 다른 문서에서 공유될 가능성이 있기 때문이다. 그러므로 우리는 또한 각각 Ign F1 and Ign AUC라는 training 및 dev/test sets에서 공유하는 relational facts를 제외한 F1 및 AUC scores를 보고한다.

**![Screen Shot 2021-06-21 at 11.33.24 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-21 at 11.33.24 PM.png)**

**Model Performance.**  Table 4는  supervised와 weakly supervised settings에서 다음과 같은 실험 결과를 보여준다.

1. human-annotated data로 훈련된 모델은 일반적으로 distantly supervised data에 대해 훈련된 모델보다 성능이 우수하다. 이는 distant supervision을 통해 large-scale distantly supervised data를 쉽게 얻을 수 있지만 wrong-labeling problem은 RE 시스템의 성능을 해칠 수 있기 때문에 weakly supervised setting을 더 어려운 시나리오로 만들기 때문이다.
2. 흥미로운 예외는 distantly supervised data에 대해 훈련된 LSTM, BiLSTM 및 Context-Aware은 human-annotated data에 대해 훈련된 것과 유사한 F1 점수를 달성하지만 다른 metrics에서는 훨씬 낮은 점수를 달성한다는 것으로, 훈련과 개발/테스트 세트 사이의 중복된 엔티티 쌍이 실제로 evaluation biases을 일으킨다는 것을 나타낸다. 따라서 Ign F1과 Ign AUC를 보고하는 것이 필수적이다.
3. 풍부한 contextual information를 활용하는 모델은 일반적으로 더 나은 성능을 달성한다. LSTM과 BiL-STM은 CNN보다 성능이 뛰어나 문서 수준 RE에서 long-dependency semantics 모델링의 효과를 나타낸다. Context-Aware는 경쟁력 있는 성능을 달성하지만 다른 neural models를 크게 능가할 수는 없다. 이는 문서 수준 RE에서 multiple relations의 연관성을 고려하는 것이 유익하지만, 현재 모델은 inter-relation information를 잘 활용할 수 없다는 것을 나타낸다.

**![Screen Shot 2021-06-22 at 12.17.27 AM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-22 at 12.17.27 AM.png)**

**Human Performance.**   Document-level RE(DocRED) 작업에 대한 human performance를 평가하기 위해, 우리는 테스트 세트에서 무작위로 100개의 문서를 샘플링하고 추가 crowd-workers에게 relation instances와 supporting evidence를 식별하도록 요청한다. Sec. 2.1과 동일한 방법으로 확인된 relation instances는 이를 지원하기 위해 crowd- workers에게 권장된다. Sec. 2.1에서 수집된 original annotation results는 실측 자료(ground truth)로 사용된다. 우리는 또한 공동으로 relation instances와 supporting evidence를 식별하는 또다른 하위 작업을 제안하고, 파이프라인 모델 또한 설계한다.  Table 5는 RE model과  human의 성능을 보여준다. Humans는 document-level RE task (RE)와 relation and supporting evidence 를 공동으로 식별하는 task (RE+Sup) 모두에서 경쟁적인 결과를 달성하며, 이는 DocRED와 annotator간 합의의 상한 성능이 모두 상대적으로 높다는 것을 나타낸다. 또한, RE 모델의 전체 성능은 인간 성능보다 현저히 낮으며, 이는 document-level RE가 어려운 작업임을 나타내며, 충분한 개선 기회를 시사한다.  서로 다른 types의 뒷받침 증거로부터 정보를 합성하는 어려움을 조사하기 위해, 우리는 개발 중인 12,332개의 관계 인스턴스를 세 개의 분리된 하위 세트로 분할한다.

**Performance v.s. Supporting Evidence Types.**  Document-level RE는 multiple supporting sentences의 정보를 합성해야 한다. 서로 다른 유형의 supporting evidence로부터 정보를 합성하는 어려움을 조사하기 위해, 우리는 development set의 12,332개의 relation instances를 세 개의 분리된 하위 집합으로 나눈다.

1. 하나의 supporting sentence인 6,115개의 relation instances(*single* 로 표시)
2. multiple supporting sentences과 entity pair가 있는 1,062개의 relation instances가 적어도 하나의 supporting sentence에서 공존한다. (*mix* 로 표시)
3. multiple supporting sentences와 entity pair가 있는 4,668개의 relation instances는 어떤 supporting sentence에서도 공존하지 않으며, 이는 multiple supporting sentences에서만 추출할 수 있음을 의미한다. (*multiple* 로 표시)

모델이 잘못된 relation를 예측할 때, 우리는 어떤 문장이 supporting evidence로 사용되었는지 알 수 없으므로 예측된 relation instance를 앞에서 언급한 하위 집합으로 분류할 수 없고 precision를 계산할 수 없다는 점에 유의해야 한다. 따라서, 우리는 *single*의 경우 51.1%, *mix*의 경우 49.4%, *multiple*의 경우 46.6%인 각 부분 집합에 대한 RE 모델의 recall만 보고한다. 이는 *mix*의 multiple supporting sentences가 보완적 정보(complementary information)를 제공할 수 있지만 풍부한 글로벌 정보(rich global information)를 효과적으로 합성하는 것이 어렵다는 것을 나타낸다. 또한, *multiple*의 저조한 성능은 RE 모델이 여전히 문장 간 관계를 추출하는 데 어려움을 겪고 있음을 시사한다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-22 at 1.46.03 AM.png" alt="Screen Shot 2021-06-22 at 1.46.03 AM" style="zoom:50%;" />

**Feature Ablations.**  우리는 entity types, coreference information  및 엔티티 간 relative distance(Eq.1)를 포함하여 문서 수준 RE에서 서로 다른 features의 기여도를 조사하기 위해 BiLSTM 모델에 대한 기능 절제 연구(feature ablation studies)를 수행한다. Table 6은 앞에서 언급한 features가 모두 성능에 기여한다는 것을 보여준다. 구체적으로, entity types는 실행 가능한 relation types에 대한 제약으로 인해 가장 많이 기여한다.  multiple named entity mentions의 정보를 합성하기 위해 Coreference information와 엔티티 간 상대적 거리도 중요하다. 이는 RE 시스템이 문서 수준에서 풍부한 정보를 활용하는 것이 중요하다는 것을 나타낸다.

**Supporting Evidence Prediction.**  우리는 relation instance에 대한 supporting evidence를 예측하는 새로운 작업을 제안한다. 한편, 증거를 공동으로 예측하는 것은 더 나은 설명 가능성을 제공한다. 반면에, text로부터 supporting evidence를 식별하고 relational facts을 추론하는 것은 자연스럽게 상호 강화 가능성이 있는 이중 작업이다. 우리는 두 가지 supporting evidence prediction methods을 고안한다.

1. Heuristic predictor.  우리는 head나 tail entity를 포함하는 모든 문장을 supporting evidence로 간주하는 간단한 heuristic-based model을 구현한다.
2. Neural predictor. 우리는 또한 neural supporting evidence predictor를 설계한다.

엔티티 쌍과 predicted relation이 주어지면, 문장은 먼저 word embeddings과 position embeddings의 concatenation에 의해 input representations으로 변환된 다음 contextual representations을 위해 BiLSTM 인코더로 입력된다.  Yang 등(2018)에서 영감을 받아, 우리는 trainable relation embedding과 첫 번째 및 마지막 위치의 BiLSTM output을 연결하여 sentence’s representation을 얻는데, 이는 문장이 주어진 relation instance에 대한 supporting evidence로 채택되는지 여부를 예측하는 데 사용된다. Table 7에서 볼 수 있듯이, neural predictor는 supporting evidence를 예측하는 데 있어 heuristic-based baseline을 크게 능가하며, 이는 joint relation 및 supporting evidence prediction에서 RE 모델의 잠재력을 나타낸다.

![Screen Shot 2021-06-22 at 1.57.32 AM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-22 at 1.57.42 AM.png)

**Discussion.**   우리는 위의 실험 결과와 분석으로부터 document-level RE가 sentence-level RE보다 더 어렵고 RE 모델과 인간의 성능 사이의 차이를 좁히기 위한 집중적인 노력이 필요하다는 결론을 내릴 수 있다. 우리는 다음과 같은 연구 방향을 따를 가치가 있다고 믿는다.

1. reasoning을 명시적으로(explicitly) 고려한 모델 탐색
2. 문장 간의 정보 수집 및 합성을 위한 보다 표현적인 모델 아키텍처 설계
3. distantly supervised data를 활용하여 문서 수준 RE의 성능을 개선

## 6. Related Work

최근 몇 년 동안 다양한 데이터 세트가 RE용으로 구성되었으며, 이는 RE 시스템의 개발을 크게 촉진시켰다. Hendrickx et al. (2010), Doddington et al. (2004) 그리고 Walker et al. (2006) 논문이 상대적으로 제한된 relation types과 instances로 human-annotated RE datasets를 구축한다. Riedel et al. (2010)은 잘못된 레이블링 문제로 어려움을 겪는 distant supervision을 통해 일반 텍스트를 KB에 정렬하여 RE 데이터 세트를 자동으로 구성한다. Zhang et al. (2017)과 Han et al. (2018b)은 external recommendations와 human annotation을 추가로 결합하여 대규모 고품질 데이터 세트를 구축한다. 그러나 이러한 RE 데이터셋은 single sentences로 relations를 제한한다.

 문서가 문장보다 풍부한 정보를 제공하므로, sentence level에서 document level로 연구가 옮겨가는 것은 많은 영역에서 대중적인 트렌드이다. 예를 들어 document-level event extraction, fact extraction and verification, reading comprehension, sentiment classification, summarization, machine translation 등이 있다. 최근에는 몇몇 document-level RE datasets도 구성되었다.  그러나 이러한 데이터 세트는 필연적으로 잘못된 레이블링 문제를 가진 distant supervision(Quirk and Poon, 2017; Peng 등, 2017)을 통해 구성되거나 특정 도메인에서 제한된다(Li 등, 2016; Peng 등, 2017). 대조적으로, DocRED는 풍부한 정보를 가진 crowd-workers에 의해 구성되며 특정 영역에 제한되지 않으므로 general-purpose document-level RE systems을 훈련하고 평가하는 데 적합하다.

## 7. Conclusion

 sentence level에서 document level로 RE 시스템을 촉진하기 위해, 우리는 데이터 크기, 여러 문장에 대한 읽기 및 추론 요구 사항, weakly supervised document-level RE의 개발을 용이하게 하기 위해 제공되는 distantly supervised data 등을 특징으로 하는 대규모 문서 수준 RE 데이터 세트인 DocRED를 제시한다. 실험 결과 human 성능이 RE baseline models보다 훨씬 높다는 것이 밝혀져 향후 개선의 기회가 충분히 있음을 시사한다.