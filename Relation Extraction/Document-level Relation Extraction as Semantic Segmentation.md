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

  대조적으로, 트랜스포머 아키텍처가 암시적으로 long-distance dependencies를 모델링 할 수 있다는 것을 고려할 때, 몇몇 연구자들은 doucument graphs를 생성하지 않고 pre-trained language models를 직접 활용한다. Wang *et al.* [2019] 은 사전 교육된 단어 임베딩으로 BERT를 사용하여 DocRED에 대한 two-step training paradigm을 제안했다. 그들은 관계 분포의 불균형을 관찰했고 더 나은 inference를 위해 relation identification 과 classification를 해체했다.