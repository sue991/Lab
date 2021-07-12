# Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling

https://github.com/wzhouad/ATLOP

## Abstract

 Document-level relation extraction (RE)은 sentence-level에 비해 새로운 과제를 제기한다. 일반적으로 하나의 문서에는 여러 개의 엔티티 쌍이 포함되어 있으며 하나의 엔티티 쌍은 여러 개의 가능한 relation과 연결된 문서에서 여러 번 발생한다. 이 논문에서, 우리는 multi-label과 multi-entity problems를 해결하기 위해 두개의 새로운 기술인 adaptive thresholding과 localized context pooling을 제안한다. Adaptive thresholding은  이전 작업의 multi-label classification에 대한 global threshold를 learnable entities-dependent threshold로 대체한다. Localized context pooling은 pre-trained language models에서 relation을 직접적으로 전달하여 relation을 결정하는 데 유용한 관련 컨텍스트를 찾는다. 우리는 세 가지  document-level RE benchmark datasets를 사용하여 실험한다: DocRED, CDR, GDA. 우리의 ATLOP(**A**daptive **T**hresholding and **L**ocalized c**O**ntext **P**ooling) 모델은 63.4의 F1 score를 달성했고, 또한 상당히 CDR과 GDA에 있는 모델보다 좋은 성능을 보였다. 

## Introduction

RE는 주어진 텍스트에서 두 엔티티 간의 관계를 식별하는 것을 목표로 하며 information extraction에 중요한 역할을 한다. 기존 연구는 주로 sentence-level relation extraction, 즉 단일 문장에서 엔티티 간의 관계를 예측하는 것에 초점을 맞춘다. 그러나, 위키백과 기사나 생물 의학 문헌의 관계적 사실과 같은 많은 양의 관계는 실제 적용에서 여러 문장으로 표현된다. 일반적으로 문서 수준 관계 추출이라고 하는 이 문제는 전체 문서에서 엔티티 간의 복잡한 상호 작용을 캡처할 수 있는 모델이 필요하다. 문장 수준의 RE와 비교하여 문서 수준의 RE는 고유한 과제를 제기한다. TACRED 및 SemEval 2010 Task 8과 같은 문장 수준의 RE 데이터셋의 경우, 문장에는 분류할 엔티티 쌍이 하나만 포함된다. 반면에 문서 레벨 RE의 경우 한 문서에 여러 엔티티 쌍이 포함되어 있으므로 이들 관계를 한 번에 분류해야 한다. RE 모델은 특정 엔티티 쌍과 관련된 컨텍스트를 사용하여 문서의 일부를 식별하고 초점을 맞추어야 한다. 또한, 문장 수준 RE의 경우 엔티티 쌍당 하나의 관계가 발생하는 것과 대조적으로 문서 수준 RE의 뚜렷한 관계와 관련된 문서에서 한 개의 엔티티 쌍이 여러 번 발생할 수 있다. 문서 수준 관계 추출의 이러한 multi-entity(문서에서 분류할 다중 엔티티 쌍) 및 multi-label(특정 엔티티 쌍에 대한 다중 관계 유형) 속성은 문장 수준 관계 추출보다 어렵다. Figure 1은 DocRED 데이터 세트의 예를 보여준다.

![Screen Shot 2021-07-12 at 8.49.12 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-07-12 at 8.49.12 PM.png)

이 Task는 엔티티 쌍(색상으로 강조 표시됨)의 relation types을 분류하는 것이다. 특정 개체 쌍(*John Stanistreet, Bendigo*)의 경우, 첫 두 문장과 마지막 문장에 의해 두 개의 relation *place of birth* 와  *place of death* 가 표현된다. 다른 문장은 이 엔티티 쌍에 대해 부적절한 정보를 포함하고 있다. 

  multi-entity problem을 해결하기 위해 대부분의 최신 접근 방식은 종속 구조, 휴리스틱 또는 구조화된 주의를 기울이는 문서 그래프를 생성한 다음 그래프 신경 모델을 사용하여 추론을 수행합니다.

  