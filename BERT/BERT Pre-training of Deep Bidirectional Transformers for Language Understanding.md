# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Abstract

  새로운 language representation model인 **BERT**(**B**idirectional **E**ncoder **R**epresentation from **T**ransformers)를 소개한다. 최근 language representation models와 달리, BERT는 모든 레이어에서 왼쪽과 오른쪽 context 공동으로 조절하여 unlabeled text로부터 deep bidirectional representations을 pre-train하도록 고안되었다. 결과적으로, pre-trained BERT 모델은 단 하나의 추가 output layer로 미세 조정되어 실질적인 작업별 아키텍처 수정 없이 질문 답변 및 언어 추론과 같은 광범위한 작업에 대한 최첨단 모델을 생성할 수 있다.

  BERT는 개념적으로 간단하고 경험적으로 강력하다. 11가지 자연어 처리 프로세싱 task에 대한 새로운 결과를 얻었다.(다양한 score에서 성능 향상)

## 1. Introduction

 언어 모델 pre-training은 많은 자연 언어 처리 작업 개선에 효과적인 것으로 나타났다. 여기에는 자연어 추론(natural language inference)과 paraphrasing과 같은 sentence-level task가 포함되는데, 그것들은 NER, question answering과 같은 token-level task뿐만 아니라 문장 사이의 관계를 전체적으로 분석함으로써 예측하는 것을 목표하고 있다.

  pre-trained language representations을 downstream tasks에 적용하기 위해 두가지 *feature-based*와 *fine-tuning* 두가지 작업이 있다. ELMo와 같은 feature-based approach는 pre-trained representations을 additional features로 포함하는 task-specific architectures를 사용한다. Generative Pre-trained Transformer(OpenAIGPT)와 같은 fine-tuning approach는 최소한의 task-specific parameters를 도입하고  모든 pre-trained parameters를 미세 조정함으로써 다운스트림 tasks에 대해 훈련된다. 두 접근법은 pre-training동안 같은 objective function을 공유하는데, 그들은 general language representations를 학습하는데 unidirectional language models을 사용한다.

  우리는 현재 기술이 특히 fine-tuning 접근법과 같은 pre-trained representations의 power를 제한하고 있다고 주장한다. 주된 limitation은 standard language models가 unidirectional이라는 것이고, 이것이 pre-training동안 사용될 수 있는 architectures의 선택을 제한한다. 예를 들어, OpenAI GPT에서, 저자는 left-to- right architecture를 사용하는데, 그것은 모든 token이 트랜스포머의 self-attention layers에 있는 이전 토큰에만 사용될 수 있다는 것이다. 그러한 제한은 문장 수준 작업에 대해 차선책이며, 질문 답변과 같은 토큰 수준 작업에 미세 조정 기반 접근방식을 적용할 때 매우 해로울 수 있으며, 양쪽 방향의 맥락을 통합하는 것이 중요한 경우이다.

  이 논문에서, 우리는 BERT를 제안하면서 fine-tuning based approaches를 향상한다. BERT는 Cloze 작업에 영감을 받아 "masked language model"(MLM) 사전 교육 목표를 사용하여 이전에 언급한 단방향성 제약을 완화한다. Masked language model은 input으로부터 몇몇 토큰을 랜덤하게 마스크하며, 목표는 마스크된 단어의 원래 vocabulary ID를 context에 따라서만 예측하는 것이다. Left-to- right language model pre-training과 달리, MLM 목표를 통해 좌측 및 우측 컨텍스트를 퓨전하여 deep bidirectional Transformer를 사전 교육할 수 있다. 마스크 언어 모델 외에도, 텍스트 쌍 표현을 공동으로 사전 교육하는 “next sentence prediction” task를 사용한다. 본 논문의 기고문은 다음과 같다.

- 우리는 language representations을 위한 bidirectional pre-training의 중요성을 증명한다. pre-training에 unidirectional language models를 사용한 것과 달리, BERT는 마스크 언어 모델을 사용하여 pre-trained deep bidirectional representations를 활성화한다. 이것은 또한 독립적으로 훈련된 left-to-right와 right-to-left LMs의 shallow concatenation을 사용한 것과 대조된다.
- 우리는 pre-trained representations가 많은 고도로 설계된 test-specific architectures의 필요성을 줄여준다는 것을 보여준다. BERT는 많은 작업별 아키텍처를 능가하는 대규모 문장 레벨 및 토큰 레벨 작업 제품군에서 최첨단 성능을 달성하는 최초의 미세 조정 기반 표현 모델이다. 
- BERT는 11개의 NLP 작업에 대해 최첨단 기술을 발전시킨다. 코드 및 사전 교육된 모델은 [다음](https://github.com/ google-research/bert)에서 확인할 수 있다. 

## 2. Related Work

  pre-training general language representations의 오랜 역사가 있으며, 이 섹션에서 가장 널리 사용되는 접근 방식을 간략하게 검토한다.

### 2.1 Unsupervised Feature-based Approaches

  non-neural과 neural methods를 포함하여 수 십년동안 광범위하게 적용가능한 words representations를 학습하는 것은 활발한 연구 분야였다. Pre-trained word embeddings은 현대 NLP 시스템의 필수적인 부분이며 scratch로부터 배운 임베딩에 비해 상당히 향상된 기능을 제공한다. Word embedding을 pre-train하기 위해, left-to-right language modeling objectives를 사용했으며, left and right context에서 정확하지 않은 단어와 구별하기 위한 목표도 사용되었다. 이러한 접근방식은 문장 임베딩(Kiros et al., 2015; Logeswaran and Lee, 2018) 또는 paragraph embeddings(Le 및 Mikolov, 2014)과 같은 세분화된 것으로 일반화되었다. sentence representations를 학습하기 위해, 이전 work는  rank candidate next sentence, 이전 문장의representation이 주어진 다음 sentence words의 left-to-right generation, 또는 denoising auto- encoder derived objectives하기 위해 objectives를 사용해왓다.

  ELMo와 이전 모델은 기존 단어 임베딩 연구를 다른 차원으로 일반화한다. 그들은 left-to-right 과 right-to-left language model로부터 *context-sensitive* features를 추출한다.각 토큰의 contextual representation은 left-to-right과 right-to-left representations의 concatenation이다. contextual word embeddings과 존재하는 task-specific architectures를 통합할 때, ELMo는 ques- tion answering, sentiment analysis, NER을 포함하는 몇몇 주요 NLP benchmarks에 대해 최신 기술을 발전시킨다. Melamud et al. (2016)은 LSTM을 사용하여 좌우 문맥에서 하나의 단어를 예측하는 과제를 통해 contextual representations을 학습할 것을 제안했다. ELMo와 비슷하게, 그들의 모델은 feature-based이고 bidirectional이 낮다. Fedus et al. (2018)는 cloze task가 text generation model의 robustness를 향상시키는데 사용될 수 있는 것을 보여준다.

### 2.2 Unsupervised Fine-tuning Approaches

  Feature-based approaches와 마찬가지로, 첫 번째 접근방식은 라벨이 부착되지 않은 텍스트에서 사전 학습된 word embedding parameters만 이 방향으로 작동합니다. 보다 최근에는contextual token representations을 생성하는 문장 또는 문서 encoder가 라벨이 부착되지 않은 텍스트에서 사전 훈련되고 감독되는 다운스트림 작업에 맞게 미세 조정되었다. 이러한 접근 방식의 이점은 처음부터 학습 될 때 파라미터가 많이 필요하지 않다는 것이다. 적어도 부분적으로는 이런 장점 때문에, OpenAI GPT는 GLUE 벤치마크에서 많은 문장 수준의 task에 대해 이전에 최첨단 결과를 달성했다. Left-to-right language modeling과 auto-encoder objectives는 몇몇 모델에서 pre-training에 사용되었다. 

### 2.3 Transfer Learning from Supervised Data

 자연어 추론(Conneau et al., 2017) 및 기계 번역(McCann et al., 2017)과 같이 대규모 데이터 세트를 사용하는 감독된 작업으로부터의 효과적인 이전을 보여주는 연구도 있다. Computer Vision 연구는 large pre-trained model로부터 transfer learning의 중요성을 입증해왔는데, 그것은 ImageNet으로 pre-traine된 모델을 fine-tuning하는 것이 효과적인 방식이라는 것이다.

## BERT

  이 섹션에서 BERT와 BERT의 세부 구현을 소개한다. 우리의 framework에는 *pre-training*과 *fine-tuning* 두 단계가 있다. pre-training동안, 모델은 다양한 pre-training task를 통해 unlabeled data에 대해 훈련받는다. Fine-tuning을 위해, BERT 모델은 먼저 pre-trained parameters로 초기화(initialize)되고, 모든 파라미터가 downstream task로부터 labeled data를 사용하여 fine-tuned된다. 각 다운스트림 작업에는 사전 교육된 동일한 매개 변수로 초기화되더라도 별도의 미세 조정된 모델이 있다. Figure 1의 question-answering example은 이 섹션의 실행 사례로 사용된다.

![Screen Shot 2021-06-27 at 1.53.32 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-27 at 1.53.32 PM.png)



  BERT의 구별되는 특징은 여러 작업에 걸친 이것의 unified(통일된) architecture이다. Pre-trained architecture와 final downstream achitecture 사이에는 최소한의 차이점이 있다.

### Model Architecture

  BERT의 모델 아키텍처는  Vaswani et al. (2017)에서 설명된 original implementation에 기반한 multi-layer bidirectional Transformer encoder아고 `tensor2tensor` library에 릴리즈 된다. Transformers의 사용이 보편화되고 구현 방식이 원본과 거의 동일하기 때문에, 모델 아키텍쳐에 대한 자세한 설명은 생략하고 독자들에게 Vaswani et al. (2017)와 "The Annotated Transformer" 같은 훌륭한 가이드라인을 참조할 것이다.

  이 연구에서, 우리는 layer 수(Transformer blocks) L, hidden size H, self-attention head 수 A를 나타낸다. 우리는 주로 두 가지 모델 크기에 대한 결과를 보고한다.
$$
BERT_{BASE} : L=12, H=768, A=12, \mbox{Total Parameters=110M} \\
BERT_{LARGE} : L=24, H=1024, A=16, \mbox{Total Parameters=340M}
$$
BERT_BASE는 purpose를 비교하기 위해 OpenAI GPT와 같은 model size를 갖는다. 그러나 결정적으로 BERT Transformer는 bidirectional self-attention을 사용하는 반면, GPT Transformer는 모든 토큰이 왼쪽에 있는 컨텍스트에만 참석할 수 있는 제한된 self-attention을 사용한다.

### Input/Output Representations

  BERT가 다양한 다운스트림 작업을 처리하도록 하기 위해, 입력 표현은 단일 문장과 문장 쌍(예: < Question, Answer >) 모두를 하나의 토큰 시퀀스로 명확하게 나타낼 수 있다. 본 연구에서 "문장"은 실제 언어적 문장이 아닌 연속적인 텍스트의 임의의 범위가 될 수 있다. "시퀀스(sequence)"는 BERT에 대한 입력 토큰 시퀀스를 의미하며, 한 문장 또는 두 문장이 함께 포함될 수 있다. 우린느 30,000 token vocabulary를 가진 WordPiece embeddings을 사용한다. 모든 시퀀스의 첫번째 토큰은 항상 special classification token(`[CLS]`)이다. 이 토큰에 상응하는 마지막 hidden state는 classification tasks의 aggregate sequence representation으로 사용된다. Sentence pair는 single sequence로 묶인다. 우리는 두가지 방법으로 문장을 구분한다. 먼저, 우리는 그것들을 special token(`[sep]`)으로 나눈다. 두번째, 우리는 학습된 임베딩을 문장 `A`에 속하는지 문장`B`에 속하는지 여부를 나타내는 모든 토큰에 추가한다. Figure 1에서 보이는것 처럼, 우리는 input Embedding을 `E`로, special `[CLS]`토큰의 final hidden vector 를 C ∈ R^H, i-th input token의 final hidden vector를 T_i ∈ R^H로 표현한다. 주어진 토큰의 경우 해당 토큰, 세그먼트 및 위치 임베딩을 합산하여input representation을 구성한다. 이 구성은 Figure 2에서 볼 수 있다.

![Screen Shot 2021-06-28 at 1.31.42 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-28 at 1.31.42 PM.png)



### 3.1 Pre-training BERT

 Peters et al. (2018a)과 Radford et al.(2018)과 달리, 우리는 BERT를 pre-traing하는데 전통적인 left-to-right 또는 right-to-left language model을 사용하지 않는다. 대신에 우리는 이번 섹션에서 설명되는 두개의 unsupervised tasks를 사용하여 BERT를 pre-train한다. 이 단계는 Figure 1의 왼쪽 부분에서 보여진다.

### Task #1 : Masked LM

 직관적으로 deep bidirectional model이 left-to-right model 또는 shallow보다 더 파워풀하다는 것은 합리적이다. 불행히도, standard conditional language model은 bidirectional conditioning이 각 단어가 간접적으로  “see itself” 할 수 있고 mule-layered context에서 target word를 하찮게 예측할 수 있기 때  오직 left-to-right 또는 right-to-left로 훈련될 수 있다. deep bidirectional representation을 훈련하기 위해 우리는 input token의 일부를 랜덤하게 마스킹한 다음 마스킹된 토큰을 예측한다. 문헌에서는 종종 *Cloze* task라고 언급되지만, 우리는 이 절차를 "masked LM"(MLM)이라고 한다. 이 경우에, mask token에 상응하는 final hidden vectors는 standard ML에서처럼 vocabulary를 통해 output softmax로 공급된다. 모든 실험에서 우리는 각 시퀀스에서 모든 WordPiece 토큰의 15%를 무작위로 마스킹한다. denoising auto-encoders와 대조적으로, 우리는 전체 인풋을 재구성하지 않고 마스킹된 단어만 예측한다.

  비록 이것이 우리가 bidirectional pretrained model을 획득할 수 있도록 하지만, 단점은 fine-tuning동안 `[MASK]` 토큰이 나타나지 않기 때문에 pre-training과 fine-tuning간에 불일치가 발생한다는 것이다. 이를 완화하기 위해 "masked" 단어를 항상 실제 `[MASK]` 토큰으로 대체하는 것은 아니다.  Training data generator는 토큰 위치의 15%를 무작위로 선택하여 예측힌다. 만약 i-th token을 선택했다면, i-th token을 (1) 시간의 80% `[MASK]` token (2) 시간의 10% 랜덤 토큰 (3) 시간의 10% unchanged i-th token 으로 바꾼다. 그다음 T_i는 cross entropy loss와 함께 original token을 예측하는데 사용될것이다. 우리는 부록 C.2.에서 이 절차의 변화를 비교한다.

### Task #2: Next Sentence Prediction (NSP)

​	Question Answering(QA), Natural Language Inference (NLI)와 같은 많은 중요한  다운스트림 tasks는 language modeling에 의해 직접적으로 포착되지 않는 두 문장들 간의 *relationship*을 이해하는데 기반을 둔다. sentence relationship을 이해하는 모델을 훈련하기 위해, 우리는 monolingual corpus에서 하찮게 생성될 수 있는 이진화된 *next sentence prediction* task를 위해 pre-train한다. 구체적으로 각 pre-training 예시에 대해 문장 A와 B를 선택할 때, 시간의 50%는 A(labeled as `IsNext`)뒤에 오는 실제 next sentence이고, 50%는 corpus(labeled as `NotNext`)의 random sentence이다. Figure 1 에서 보는 것 처럼, `C`는 next sentence prediction(NSP)에 사용된다.   단순함에도 불구하고, 우리는 Section 5.1에서 이 task를 위한 pre-training이 QA와 NLI모두에서 매우 유용하다는 것을 보여준다. NSP 과제는 저나이트 외 연구진(2017년), 로그스워란 및 리(2018년)에 사용된 대표 학습 목표와 밀접한 관련이 있다. 그러나 이전 연구에서 오직 sentence embeddings만 down-stream task로 전송되고 ,  BERT는 모든 매개변수를 전송하여 최종 model parameters를 초기화한다.

**Pre-training data**  pre-training 절차는 language model pre-training에서 기존 문헌을 따른다. 