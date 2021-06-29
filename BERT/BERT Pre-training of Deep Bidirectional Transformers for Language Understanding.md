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

**Pre-training data**  pre-training 절차는 language model pre-training에서 기존 문헌을 따른다.  pre-training corpus의 경우 BooksCorpus (800M words) 와 English Wikipedia(2,500M words)를 사용한다. 위키피디아의 경우 오직 text 구절만 추출하고 list, table, headers는 무시한다.  Long contiguous sequences를 추출하기 위해 Billion Word Benchmark와 같은 셔플된 sentence-lavel corpus보다는 document-level corpus를 사용하는 것이 중요하다. 

### 3.2 Fine-tuning BERT

​	트랜스포머의 self-attention mechanism은 BERT가 single text혹은 text pairs를 포함하든 간에 적절한 인풋 아웃풋을 바꿈으로서 많은 다운스트림 task를 모델링할 수 있또록 하기 때문에 fine-tuning은 간단하다. Text pairs를 포함하는 어플리케이션의 경우, 공동 패턴은  Parikh et al. (2016); Seo et al. (2017)과 같이 bidirectional cross attention을 적용하기 전에 텍스트 쌍을 독립적으로 인코딩 하는 것이다. 대신 BERT는 두 단계를 통합하기 위해 self-attention mechanism을 사용한다. self-attention을 갖고 concatenated된 text pair을 인코딩하면 두 문장 사이에 *bidirectional* cross attention이 포함되기 때문이다. 각 task에서 간단히 task별 inputs과 outputs를 BERT에 연결하고 모든 parameters를 종단 간 fine-tune한다. Input에서 pre-training의 문장 A와 문장 B는 (1) paraphrasing의 sentence pairs, (2) 수반되는  hypothesis-premise pairs, (3) question answering에서 question-passage pairs, (4) text classificatino 또는 sequence tagging에서 악화된(degenerate) text-∅ pair와 유사하다. Output에서는, token representation은 sequence tagging이나 question answering과 같은 token-level tasks를 위한 output layer로 공급되고, `[CLS]` representation은 entailment또는 sentiment analysis과 같은 classification을 위한 output layer로 공급된다.

  Pre-training과 비교해서, fine-tuning은 상대적으로 저렴하다. 논문의 모든 결과는 동일한  pre-trained model에서 시작하여 많아봤자 1시간의 단일 클라우드 TPU, 몇 시간의 GPU에서 복제할 수 있다. 섹션 4의 해당 하위 섹션에서 task별 세부 사항을 설명한다. 더 자세한 내용은 A.5.에서 확인할 수 있다.

## 4. Experiments

여기서 우리는 11NLP tasks에서의 BERT fine-tuning 결과를 보여준다.

### 4.1 GLUE

  General Language Understanding Evaluation (GLUE) benchmark는 다양한  natural language understanding tasks의 모음이다. GLUE datasets의 자세한 설명은 B.1을 봐라.

  GLUE를 fine-tune하기 위해, Section 3에 설명한대로 input sequence (single sentence or sentence pairs를 위한)를 나타내고, 첫번째 인풋 토큰(`[CLS]`)에 해당하는 final hidden vector C ∈ R^H를 aggregate representation으로 사용한다. Fine-tuning동안 도입된 새로운 파라미터는 classification layer weights W ∈ R^{KxH}인데, 이때 K는 # of labels이다. 우리는 C,W로 standard classification loss를 계산한다.
$$
log(softmax(CW^T))
$$
  32의 batch size를 사용하며 모든 GLUE tasks에 대해 3 epoch동안 fine-tune 한다. 각 task에 대해, Dev set에서 가장 fine-tuning 된 learning rate(5e-5, 4e-5, 3e-5, and 2e-5)를 선택했다. 추가적으로  BERT_LARGE 를 위해 fine- tuning이 때때로 small datasets에서 불안정하다는 것을 발견했고, 몇번 무작위로 재시작하고 Dev set에서 best model을 선택했다. Random restarts에서는 동일한 pre-trained checkpoint를 사용하지만 fine-tuning data shuffling과 classifier layer initialization을 다르게 수행한다.

![Screen Shot 2021-06-28 at 5.38.14 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-28 at 5.38.14 PM.png)  

결과는 Table 1처럼 보여진다. BERT_BASE와 BERT_LARGE 모두 이전 기술보다 각각 평균 정확도가 4.5%, 7.0% 향상되어 모든 작업에서 모든 시스템을 상당한 차이로 능가한다. BERT_BASE와 OpenAI GPT는 attention masking과 별개로 거의 모델 아키텍쳐 부분에서 거의 동일하다는 것을 주목하자. 가장 크고 가장 널리 보고되는 GLUE task인 MNLI, BERT는 절대 정확도가 4.6% 향상되었다. 공식 GLUE leaderboard에서, BERT_LARGE는 72.8점을 획득한 OpenAI GPT와 비교해서 80.5 score를 얻었다(작성일 기준).

  BERT_LARGE는 모든 작업, 특히 training data가 거의 없는 작업에서 BERT_BASE를 크게 능가한다. 모델 사이즈의 효과는 섹션 5.2에서 더 자세히 다뤄진다.

### 4.2 SQuAD v1.1

  The Stanford Question Answering Dataset (SQuAD v1.1)은 10만 개의 crowd- sourced question/answer pairs를 모은 것이다. 정답을 포함하는 위키피디아의 문구와 구절이 주어졌을때,  과제는 해당 구절의 정답 텍스트 범위를 예측하는 것이다.  Figure 1에서 보는것 처럼, question answering task에서, input question과 passage를 A embedding을 사용한 question과 B embedding을 사용한 passage와 함께 single packed sequence로 표현한다. 우리는 fine-tuning동안 오직 start vector S ∈ R^H와 end vector E ∈ R^H를 도입한다. 단어 i가 answer span의 start일 확률은 paragraph에 있는 모든 단에 대한 softmax에 이어 T_i와 S의 dot product로 계산된다.
$$
P_i = \frac{e^{S·T_i}}{\sum_je^{S·T_j}}
$$
유사한 공식이 answer span 끝에 사용된다. 위치 i,j에 대한  candidate span의 점수는  S ·Ti + S ·Tj로 정의되고, j >= i인 최대 scoring span은 prediction으로 사용된다. Training objective는 정확한 start와 end position의 log-likelihoods의 합이다. 우리는 5e-5의 lr, 32 batch size로 3 epoch동안 fine-tune한다.

![Screen Shot 2021-06-28 at 6.27.46 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-28 at 6.27.46 PM.png)

  Table 2는 최상위 리더보드 항목과 상위 게시된 시스템의 결과를 보여준다. SQuAD 리더보드의 상위 결과에는 public system descriptions available이 없으며 시스템 교육 시 공개 데이터를 사용할 수 있습니다. 그러므로 SQuAD를 fine-tuning 하기 전에 먼저 TriviaQA를 fine-tuning하여 우리의 시스템에서 보통의 data augmentation을 사용한다. 

 최고 성능 시스템은 앙상블 시 최고 리더보드 시스템보다 +1.5F1 그리고 단일 시스템으로는 +1.3F1만큼 성능이 뛰어나다. 사실, 당사의 단일 BERT 모델은 F1 score 면에서 상위 앙상블 시스템을 능가한다. TriviaQA fine-tuning data없이, 오직 0.1-0.4 F1만 읽고, 여전히 모든 존재하는 시스템보다 훨씬 능가한다.

### 4.3 SQuAD v2.0

  SQuAD 2.0 작업은 제공된 단락에 짧은 답이 존재하지 않을 수 있도록 하여 SQuAD 1.1 문제 정의를 확장하여 문제를 보다 현실적으로 만든다. 

 이 작업을 위해 간단한 접근 방식을 사용하여 SQuAD v1.1 BERT 모델을 확장한다. 우리는 답이 없는 질문은 `[CLS]` 토큰에서 시작과 종료 사이에 답이 있는 것으로 취급한다. 시작 및 끝 응답 범위 위치의 확률 공간은 `[CLS]` 토큰의 위치를 포함하도록 확장된다. 예측을 위해 no-answer span의 점수(s_null = S·C + E·C)를 best non-null span와 비교한다.
$$
\hat s_{i,j} = max_{j>=i} S·T + E·T_j  \mbox{  : best non-null span}
$$


 F1을 최대화하기 위한 threshold  τ가 dev set에서 선택되고, 
$$
\hat s_{i,j} > s_{null} + τ
$$


일 때 non-null answer를 예측한다. 우리는 이 모델에서 TriviaQA data를 사용하지 않는다. 우리는 lr 5e-5, batch size of 48에서 2 epoch동안 fine-tune 한다.

![Screen Shot 2021-06-28 at 6.44.32 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-28 at 6.44.32 PM.png)

이전의 leaderboard entries와 top published work와 비교한 결과는 Table 3에서 보여주는데, BERT를 하나의 구성요소로 사용하는 시스템은 제외한다. 이전의 최고 시스템 대비 +5.1 F1이 개선되었다. 

### 4.4 SWAG

  The Situations With Adversarial Generations (SWAG) dataset 기초적인 상식 추론을 평가하는 113k 문장 쌍 완성 예가 포함되어 있다. 주어진 문장에서 과제는 네 가지 선택 중에서 가장 그럴듯한 연속성을 선택하는 것이다.

   SWAG dataset에서 fine-tuning할 때, 우리는 주어진 문장(문장 A)과 가능한 연속(문장 B)의 연결을 각각 포함하는 4개의 입력 시퀀스를 구성한다. 도입된 유일한 작업별 매개변수는 [CLS] 토큰 표현 C를 가진 도트곱이 소프트맥스 계층으로 정규화된 각 선택에 대한 점수를 나타내는 벡터이다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-28 at 7.19.49 PM.png" alt="Screen Shot 2021-06-28 at 7.19.49 PM" style="zoom:50%;" />

우리는 모델을  lr 2e-5, batch size 16으로 3 epoch동안 fine-tune한다. 결과는 Table 4에 보여진다. BERT_LARGE는 저자의 기본 ESIM+ELMo 시스템보다 27.1%, OpenAI GPT 보다 8.3% 능가하는 성능을 보였다.

## Ablation Studies

  이 섹션에서, 우리는 상대적인 중요성을 더 잘 이해하기 위해 BERT의 여러 측면에 대해 ablation experiments를 수행한다. 추가적인 ablation studies가 부록 C에서 보여진다.

### 5.1 Effect of Pre-training Tasks

 BERT_BASE와 정확히 동일한 pre-training 데이터, 미세 조정 체계 및 하이퍼 매개변수를 사용하여 두 가지 사전 교육 목표를 평가함으로써 BERT의 심층적인 양방향성이 얼마나 중요한지 보여준다. 

**NO NSP** : "masked ML"(MLM)을 사용하여 훈련되지만 "next sentence prediction(NSP)" task가 없는 bidirectional model.

**LTR & No NSP** : MLM이 아닌 표준 Left-to-Right(LTR)을 사용하여 훈련하는 left-context-only model. Left-only constraint가 fine-tuning에 적용되는데, 제거하면 다운스트림 성능이 저하되는  pre-train/fine-tune 불일치가 발생했기 때문이다. 게다가 이 모델은  NSP task없이 pre-trained되었다. 이것은 더 큰 training dataset, 우리의 input representation, 우리의 fine-tuning scheme 를 사용하면 직접적으로 OpenAI GPT와 비교가능하다. 

![Screen Shot 2021-06-28 at 7.32.50 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-28 at 7.32.50 PM.png)

 우리는 먼저 NSP task의 영향을 실험한다. Table 5에서, NSP를 없애는 것은 QNLI, MNLI, and SQuAD 1.1 성능에 상당한 저하를 준다. 다음, 우리는 "No NSP" , "LTR & No NSP"를 비교하여 birirectional representations training의 영향을 평가한다. LTR 모델은 모든 작업에서 MLM 모델보다 성능이 떨어지며 MRPC 및 SQuAD가 크게 저하된다. SQuAD의 경우  token-level hidden states는 right-side context가 없기 때문에 LTR 모델이 토큰 예측에서 성능이 저하된다는 것이 직관적으로 분명하다. LTR 시스템 강화를 위해 무작위로 초기화 된 BiLSTM을 상단에 추가했다. 이것은 SQuAD의 결과를 상당히 향상시켰지만, 결과는 여전히 pre-trained bidirectional models보다 낮다. BiLSTM은 GLUE task의 성능을 저하시킨다. 우리는 또한 개별 LTR과 RTL 모델을 훈련시키는 것이 가능할 것이며, ELMo가 하는 것처럼 각각의 토큰을 두 모델의 연결로 나타낼 수 있음을 인정한다. 그러나, (a) 이는 단일 양방향 모델보다 2배 더 비싸다. (2) 이는 QA와 같은 과제에서는 직관적이지 않다. RTL 모델은 질문에 대한 답을 조건화할 수 없기 때문이다. 이는 모든 계층에서 좌우 컨텍스트를 모두 사용할 수 있기 때문에 양방향 심층 모델보다 훨씬 강력하지 않다.

### 5.2 Effect of Model Size

  이 섹션이서, 우리는 fine-tuning task accuracy에서 model size의 효과를 보여주다. 우리는 다른 layer 수, hidden unit 및 attention head의 수로 여러 BERT 모델을 train했으며, 그렇지 않은 경우에는 앞에서 말한것과 같은 하이퍼 파라미터와 training 절차를 사용했다.

![Screen Shot 2021-06-29 at 2.33.54 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-29 at 2.33.54 PM.png)

  선택된 GLUE task의 결과는 Table 6에서 보여준다. 이 표에는 fine-tuning의 랜덤 restart 5회부터 평균 DevSet의 정확도가 나와있다. 우리는 더 큰 모델이 4개의 데이터셋 모두에서 정확도 향상으로 이어진 다는 것을 볼 수 있다. 심지어 3,600개의 라벨링된 training example만 있고 pre-training task와는 상당히 다른 MRPC에서도 그렇다. 또한 기존 문헌과 연관된 이미 꽤 큰 모델보다 상당한 향상을 달성할 수 있었다는 것도 놀랍다. 예를들어, Vaswani et al. (2017)에서 탐구된 가장 큰 트랜스포머는 인코터에  (L=6, H=1024, A=16)와 100M개 파라미터가 있고, 발견한 가장 큰 트랜스포터는 (L=64, H=512, A=2)와 235M개 파라미터이다. 대조적으로, BERT_BASE는 110M개 파라미터를 갖고있고 BERT_LARGE는 340M개 파라미터를 갖고있다.

 모델 크기를 들리면 기계번역 및 언어 모델링과 같은 대규모 작업이 지속적으로 개선된다는 것은 오래전부터 알려진 사실이고, 이는 Table 6에 나와 있는 보류된 데이터의 LM perplexity(당혹감, 곤란)을 통해 입증된다. 그러나 우리는 모델이 충분히 pre-trained되면 극한의 모델 사이로 확장하면 또한 매우 작은 규모의 task에서도 대규모 향상으로 이루어 진다는 것을 설득력있게 입증하는 첫번째 방법이라고 믿는다. Peters et al. (2018b)은 pre-trained된 bi-LM 크기를 2에서 4개 layer로 증가시키는 것의 다운스트림 task 영향에 대한 혼합 결과를 보여주고, Melamud et al. (2016)은 hidden dimension size를  200에서 60으로 늘리는 것이 도움되었지만, 1000개 이상에서는 더이상의 향상이 없었다고 말했다.  이러한 이전 연구는 feature based approach를 사용했는데, 우리는 모델이 직접적으로  다운스트림 task에 fine-tuned되고 오직 매우 작은 수의 무작위로 초기화된 추가적인 파라미터를 사용했을 때, 심지어 다운스크림 task가 매우 작을때에도 task-specific 모델은 더 크고, 효과적인 pre-trained representations으로부터 이익을 볼 수 있다.

 ### 5.3 Feature-based Approach with BERT

  지금까지 제시된 모든 BERT 결과는 사전 교육된 모델에 단순한 분류 계층이 추가되고 모든 매개변수가 다운스트림 작업에서 공동으로 미세 조정되는 미세 조정 방식을 사용해 왔다. 그러나 사전 교육된 모델에서 고정된 특징을 추출하는 기능 기반 접근방식은 특정한 장점을 가지고 있다. 첫째, 모든 Task가 쉽게 Transformer 인코더 아키텍처로 표현되는 것은 아니므로 task-specific model architecture가 추가로 필요하다. 두번째, 비싼 training data의 representatioin 를 pre-compute한 다음 이 표현 위에 더 저렴한 모델로 많은 실험은 할 수 있다는 주요 연산 이점이 있다.

  이 섹션에서 우리는 BERT를 CoNLL-200 Named Entity Recognition (NER) 에 적용함으로써 두 접근법을 비교한다. BERT의 인풋에서 우리는  case-preserving WordPiece model을 사용하고, 데이터에 제공된 maximal document context를 포함한다. 표준 관행에 따라 이 작업을 태깅 작업으로 공식화하지만 출력에 CRF layer를 사용하지 않는다. 우리는 첫번째 sub-token의 representation을 NER lavel set을 통해 token-level classifier에 대한 입력으로 사용한다. Fine-tuning approach를 없애기 위해 BERT의 파라미터를 fine-tuning하지 않고 하나 이상의 레이어에서 activations를 추출하여 feature-based approach를 적용한다. 이러한 상황별 임베딩은 분류 계층 이전에 무작위로 초기화된 2-layer 768-dimensional BiLSTM에 대한 입력으로 사용된다.

![Screen Shot 2021-06-29 at 3.15.44 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-29 at 3.15.44 PM.png)

  결과는 Table 7에서 보여진다. BERT_LARGE는 최첨단 방식으로 경쟁적으로 수행한다. 최상의 성능 방법은 전체 모델을 미세 조정하는 데 0.3F1밖에 뒤지지 않는 사전 교육된 Transformer의 상위 4개 숨겨진 계층의 토큰 표현을 concat한다. 이것은 BERT가 fine-tuning과 feature-based approaches 둘 다 효과적이라는 것을 보여준다.

## 6. Conclusion

  언어 모델을 통한 이전 학습으로 인한 최근의 경험적 개선은 풍부하고 감독되지 않은 사전 훈련이 많은 언어 이해 시스템의 필수적인 부분이라는 것을 보여주었다. 특히 이러한 결과를 통해 리소스가 낮은 작업도 심층적인 단방향 아키텍처를 활용할 수 있다. 우리의 주요 기여는 이 발견을 깊은 *bidirectional* architecture로 더욱 일반화 함으로써 pre-trained된 동일한 모델이 광범위한 NLP task를 성공적으로 처리할 수 있게 하는것이다.