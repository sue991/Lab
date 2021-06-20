# DocRED: A Large-Scale Document-Level Relation Extraction Dataset

## Abstract

 문서 내 multi entities는 일반적으로 복잡한 상호작용 관계를 나타내며, single entity pairs를 위해 일반적으로 intra-sentence relations에 초점을 맞춘 existing relation extraction(RE) 방법에 의해 잘 처리될 수 없다. document-level RE에 대한 연구를 가속화하기 위해, 우리는 위키피디아와 위키데이터에서 생성된 새로운 데이터 세트인 **DocRED**를 소개합니다. 이 DocRED는 세가지 특징이 있는데, 

1. DocRED는 named entities와 relations 모두를 annotate하며, plain text에서 document-level RE를 위한 가장 큰 human-annotated dataset이다.

2. DocRED는 여러 문장들을 문서에서 판독하여 entity를 추출하고 문서의 모든 정보를 합성함으로써 그들의 relation를 추론하도록 요구한다.
3. human-annotated data와 마찬가지로, DocRED가 supervised 그리고 weakly supervised scenarios에 채택될 수 있도록 하는 large-scale distantly supervised data를 제공한다.

Document-level RE의 과제를 검증하기 위해 RE를 위한 최신 연구를 구현하고 DocRED에서 이러한 방법의 철저한 평가를 수행한다. 경험적 결과는 DocRED가 기존의 RE 방법들에 도전하고 있음을 보여 주고 있으며, 이는 document-level  RE가 개방된 문제로 남아 있고 추가 노력이 필요하다는 것을 보여준다. 실험들에 대한 상세한 분석을 기반으로, 우리는 향후 연구를 위한 여러 가지 약속 방향들을 논의한다. 우리는 DocRED와 baselines 코드를 다음 [주소](https: //github.com/thunlp/DocRED)에서 공개적으로 사용할 수 있게 한다. 

## 1. Introduction

relation extraction(RE)의 과제는 large-scale knowledge graph construction에 중요한 역할을 하는 일반 텍스트에서 entities 간의 relational facts을 식별하는 것이다. 대부분의 기존 RE 작업은 sentence-level RE, 즉 single sentence에서 관계 사실을 추출하는 데 중점을 둔다.

