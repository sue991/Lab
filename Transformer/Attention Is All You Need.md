# Attention Is All You Need

## Transformer 

2021년 기준 현대의 자연어 처리 네트워크에서 핵심이 되는 눈문.

RNN, CNN등을 전혀 필요로 하지 않는다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 3.28.07 PM.png" alt="Screen Shot 2021-06-25 at 3.28.07 PM" style="zoom:50%;" />



RNN, CNN을 전혀 사용하지 않고 문장안에 포함되어있는 단어 간의 순서에 대한 정보를 알려주기 어렵다. 그래서 문장안에 포함되어있는 단어 간의 순서에 대한 정보를 알려주기 위해 Positional Encoding을 이용해 순서에 대한 정보를 주고 있다.

인코더, 디코더로 구성된다. 또한 Attention 과정을 한번이 아닌 여러 레이어에서 반복한다.

즉, 이러한 인코더가 N번만큼 중첩되어 사용되는 것이다.

## 동작 원리: Encoder

트렌스포머 이전의 전통적인 임베딩은 Input Embedding maxtrix를 이용해 단어의 개수만큼 행으로 만들고 열로 embed_diim을 사용하여 구했다.  즉, 전통적인 임베딩은 네트워크에 넣기 전 입력값들을 임베딩 형태로 표현하기 위해 사용하는 레이어.

RNN을 사용하는 것 만으로도 단어가 RNN에 들어갈 때 순서에 맞게 들어가기 때문에 자동으로 각각 hidden state값은 순서에 대한 정보를 가지게 된다. 그런데 트랜스포머는 RNN을 사용하지 않기 때문에 위치 정보를 주기 위해 위치 정보를 포함하고 있는 임베딩을 사용해야 한다.

이를 위해 트랜스포머에서는 위치에 대한 정보를 인코딩하고 있는 Positional Encoding을 사용한다. input embedding matrix와 같은 크기, 즉 같은 dimension을 가지고 있는 별도의 위치 정보를 갖고 있는 인코딩 정보를 넣어줘서 각각 element-wise로 더해줌으로써 각각 단어가 어떤 위치에 있는지에 대한 정보를 네트워크가 알 수 있도록 만들어주는 것.

임베딩이 끝난 이후 어텐션을 진행한다.

어텐션이 받는 값은 입력 문장에 대한 정보에 위치에 대한 정보도 포함되어 있는 입력값이다.

이제 이런 입력값을 받아 각각 단어의 어텐션을nul 수행하고, 인코더에서 수행되는 어텐션은 셀프-어텐션이라고 한다. 각각 단어가 서로에게  어떤 연관성을 가지고 있는지를 구하기 위해 사용한다.

![Screen Shot 2021-06-25 at 3.39.02 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 3.39.02 PM.png)

I, am, a, teacher가 각각 서로에게 어텐션 스코어를 구해서 각의 단어는 다른 어떤 단어와 높은 연관성을 가지는지 학습할 수 있다.

어텐션은 전반적인 입력문장 대한, 문맥에 대한 정보를 잘 학습할 수 있도록 만드는 것이다.

또한 여기에서 residual learning같은 테크닉에 사용한다.

대표적인 resnet에 사용하는 기법인데, 특정 레이어를 건너뛰어 복사가 된 값을 그대로 넣어주는 기법.

이렇게 건너뛰에 넣어주는 기법을 residual connection이라고 하고, 이렇게 해줌으로써 네트워크는 기존 네트워크를 입력 받으면서, 추가적으로 잔여된 부분만 학습하도록 만들기 때문에 전반적인 학습 난이도가 낮고 초기 모델 수렴속도가 높게 되고, 그렇게 때문에 전반적인 global optimal을 찾을 확률이 높기때문에 전반적인 네트워크에서 residual learning을 사용했을 때 성능이 높게 되는 것을 확인할 수 있다. 트랜스포머 또한 채택해서 성능을 높였다.

![Screen Shot 2021-06-25 at 3.44.45 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 3.44.45 PM.png)

이렇게 residual learning을 사용하고 normalize까지 진행하는 것이 인코더의 동작 과정이다.



![Screen Shot 2021-06-25 at 3.45.58 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 3.45.58 PM.png)

이 그림처럼 attention을 거치고 residual과 normalize하고, 그리고 다시 feedforward layer를 거친 다음에 residual learning, norm을 추가해서 결과적으로 하나의 인코더레이어에서 결과값을 뽑아낼 수 있다. 

이렇게 어텐션과 정규화 과정을 반복하는 방식으로 여러개의 레이어를 중첩해서 사용한다. 이때 유의할 점은 각 레이어는 서로 다른 파라미터를 가진다. 다만, 레이어에 입력되는 값과 출력되는 값의 dimension은 동일하다.

## 인코더와 디코더

![Screen Shot 2021-06-25 at 3.49.07 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 3.49.07 PM.png)

실제로는 이 그림과 같이 인코더와 디코더의 아키텍쳐를 그려볼 수있다.

인코더 레이어를 여러번 반복해서 가장 마지막에서 나온 출력값이 디코더의 입력값으로 들어간다. 이유는 디코더 파트에서는 입력 소스 문장 중에서 어떤 단어에게 가장 많은 초점을 둬야하는지 알려주기 위함이다. 다시말해 디코더도 여러 레이어로 이루어지게 되고 마지막 레이어에서 나오는 출력값이 실제 번역을 수행한 결과(출력 결과)가 된다.

디코더 또한 마찬가지로 각각 단어 정보를 받아 위의 정보의 인코딩 값을 추가한 다음 입력으로 넣게되고, 하나의 디코더 레이어에서는 2개의 어텐션을 사용하는데, 첫번째는 셀프 어텐션으로 인코더와 마찬가지로 각 단어가 서로에게 어떠한 가중치를 가지는지 구하도록 만들어서 출력되고 있는 문장에 대한 전반적인 표현이 학습될 수 있도록 만든다. 두번째 레이어에서는 인코더의 출력 정보를 어텐션하도록 만든다. 다시말해 각각 출력 단어가 인코더의 출력 정보를 받아와 사용할 수 있도록 만드는 것이다. 즉, 각각 출력되고 있는 단어가 소스 문장에서의 어떤 단어와 연관성이 있는지 말해주는 것이다. 

즉, 디코더도 입력으로 들어온 입력 디멘전과 출력 dim이 같도록 만들어 각각 디코더 레이어는 여러번 중첩해서 사용할 수 있다. 다시말해 트랜스포머에서는 마지막 인코더 레이어의 출력이 모든 디코더 레이어에 입력되는 형식으로 동작한다.

전체 레이어 개수가 4개일 경우 다음과 같다.

![Screen Shot 2021-06-25 at 3.59.10 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 3.59.10 PM.png)

일반적으로 인코더와 디코더 레이어 개수가 같게 만든다.

트랜스포머에서도 인코더와 디코더 구조를 따른다. 이때 RNN을 사용하지 않으며 인코더와 디코더를 다수 사용한다는 점이 특징이다. RNN과 다르게 위치에 대한 정보를 한꺼번에 넣어서 한번의 인코더를 거칠때마다 병렬적으로 출력값을 구해낼 수 있기 때문에 RNN을 사용했을때와 비교하여 일반적으로 계산복잡도가 낮게 형성된다. 실제로 학습을 수행할 때 입력값들을 한꺼번에 넣을 수 있기 때문에 RNN을 사용하지 않는 것이 장점인데, 다만 모델에서 출력값을 내보낼 때는 디코더 아키텍처를 여러번 사용해서 <eos> 가 나올때까지 반복하도록 만들어서 출력값을 구하도록 만든다.

## Multi-head Attention

트랜스포머에서 사용되는 각각의 어텐션은 여러개의 헤드를 가진다고 해서 멀티헤드 어텐션이라고 불린다. ![Screen Shot 2021-06-25 at 4.16.57 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 4.16.57 PM.png)

중간에 Scaled Dot-Product Attention이 사용되는데, 이건 왼쪽 그림처럼 구성된다.

어텐션 메커니즘을 이해하기 위해선 쿼리, 키, Value가 무엇인지 알아야한다.

쿼리(Query) : 무언가를 물어보는 주체. 

키(Key) : 물어보는 대상.

"i am a teacher" 라는 문장에서 I 가 self attention을 수행하여 각각 단어  i, am, a, teacher와의 연관성을 구한다고 치면 i는 쿼리가 되고 각각 단어는 key가 되는 것.  각 단어에 대한 가중치값을 구한다고 치면 각 키에 대해 attention score를 구해 오는 방식으로 동작하는 것. score를 구한 뒤 실제로 value값들과 곱하고 결과적으로 attention value를 구한다.

Q: query		K : attention을 수행할 단어들

softmax를 통해 K중 어떤 단어가 연관성이 높은지 구할 수 있음.

이런 확률값과 value를 곱해 가중치가 적용된 결과적인 attention value를  구할 수 있는 것이다. 이런 과정이 scaled Dot-Product Attention에서 수행되는 것이다.

여기에서 참고로 실제로 입력값이 들어오면 h개로 구분되는데, 즉 어떤 입력 값이 들어오면 h개의 서로다른 value, key, query로 구분될 수 있도록 만드는 것이다. 이렇게 해주는 이유는 h개의 서로 다른 attention concept을 학습하도록 만들어 더욱더 구분된 다양한 특징을 학습할 수 있도록 유도해주는 특징이 있다.

이와같이 입력으로 들어온 값은 세개로 복제가 되어 각각 V,K,Q로 들어가게 되고 , 이 V,K,Q 값들은 linear layer, 즉 행렬 곱을 수행해 h개로 구분된 각각 쿼리 쌍들을 만들어내게 되고, 이때 h는 head의 개수로 각각 서로 다른 head끼리 V,K,Q 쌍을 받아 어텐션 수행해 결과를 내보낸다.

어텐션 메커니즘의 입력, 출력 디멘션이 같아야하기 때문에 각각 헤드로부터 나온 어텐션 값들을 concat해서 붙인 뒤 linear layer를 거쳐 output 값을 내보낸다.

인코더 디코더에 사용되는 어텐션은 Key,Value를 인코더의 출력으로 사용하고, 디코더에 있는 단어가 Query가 된다.

![Screen Shot 2021-06-25 at 4.35.17 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 4.35.17 PM.png)

Multi-head Attention 레이어를 수식으로 표현한 것.

d_k : 각각 key의 dimension 

## Query, Key, Value

하나의 단어만 있다고 가정할 때 트랜스포머의 동작 원리를 알아보자.

어텐션을 위해 각 head마다 쿼리, 키, 벨류 값을 만들 필요가 있다.

![Screen Shot 2021-06-25 at 4.59.04 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 4.59.04 PM.png)

그래서 각 단어가 임베딩 차원으로 표현되고 있는 상황에 Linear layer를 거쳐 각각 Q,K,V 값을 만들 수 있다. 임베딩 차원을 d_model이라고 할 때, Q,K,V 차원은 각각 (d_model / h) 가 된다. 이 그림은 간단히 임베딩차원이 4, head가 2개라고 가정한 상황이다.

각각 Q,K,V를 구한 후 attention value를 구할 수 있다.

![Screen Shot 2021-06-25 at 5.01.43 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.01.43 PM.png)

쿼리는 각각 다른 단어들(Key)와 행렬곱을 수행해 각각 하나의 attention energy값을 구할 수 있다. 그리고 softmax에 들어가는 값의 크기 normalization해주기 위해 각각 scaling factor로 나누어준다. 이후에 softmax를 취하고 가중치를 구한다.  각각 가중치값에 value값들을 곱한 뒤 전부 더해 결과적으로 attention value 값을 구할 수 있다. 즉 마찬가지로 weighted sum을 구할 수 있다는 것이다.

이 그림을 보면 I라는 단어는 I라는 단어와 72%의 높은 연관성을 가지고 you라는 단어와는 13%의 가중치 값을 가지고 있다는 것을 나타내고 있다.

실제로 전체 문장이 한꺼번에 입력되는 행렬의 경우, 행렬 곱셈 연산을 이용해 한꺼번에 연산이 가능하다.

![Screen Shot 2021-06-25 at 5.08.24 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.08.24 PM.png)

![Screen Shot 2021-06-25 at 5.09.21 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.09.21 PM.png)

attention value값 자체는 입력되었던 Q,K,V와 동일한 차원을 가진다.



![Screen Shot 2021-06-25 at 5.23.52 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.23.52 PM.png)

또 한가지 알아두면 좋은 점은 마스크 행렬을 사용할 수 있다는 점이다. 마스크행렬은 특정한 단어는 무시할 수 있도록 하는 것이다.

attention energy가 있을 때, 같은 차원의 마스트 행렬을 만들어서 element-wise로 곱하면 어떤 단어들은 참고하지 않도록 만드는 것이다.

마스크매트릭스를 이용함으로써 특정 단어는 무시해서 어텐션을 수행하지 않도록 만들 수 있다.

![Screen Shot 2021-06-25 at 5.24.51 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.24.51 PM.png)

결과적으로 각각의 head마다 입력으로 들어온 Q,K,V와 같은 차원의 벡터를 만들어내기 때문에 각각 넣어서 attention 수행한 값들을 concat하면 맨 처음 넣었던 입력 dim과 같은 dim을 만들어낸다.

![Screen Shot 2021-06-25 at 5.29.19 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.29.19 PM.png)



### Attention 종류

![Screen Shot 2021-06-25 at 5.32.48 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.32.48 PM.png)사용되는 위치에 따라 세가지 종류의 어텐션이 사용된다.

1.  Encoder Self-Attention : 각각의 단어가 서로에게 어떠한 연관성을 가지는지 어텐션을 이용해 구하고 전체 문장에 대한 representation을 learning할 수 있도록 만드는 것이 특징
2. Masked Decoder Self-Attention : Decoder에서 self-attention을 수행할 때 각각의 출력 단어가 모든 출력 단어를 보지 않고 앞쪽에 등장했던 단어만 참고할 수 있도록 만든 것
3. Encoder-Decoder Attention : 쿼리가 디코더에 있고 키와 벨류가 인코더에 있는 상황을 의미

### Self-Attention

인코더와 디코더 모두에서 사용된다. 매번 입력 문장에서 각 단어가 다른 어떤 단어와 연관성이 높은지 구할 수 있다.![Screen Shot 2021-06-25 at 5.40.10 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.40.10 PM.png)



## Transformer

인코더 파트에서는 입력값이 들어와서 위치에 대한 정보를 반영해 준 입력을 첫번째 레이어에 넣어주게 되고, 레이어는 N번만큼 반복이 되어 중촙돼 사용이 된다. 그리고 마지막 레이어의 출력 값이 각각의 디코더 레이어에 들어가게 된다.

디코더 레이어도 N번만큼 중첩이 되고 마지막에 나온 출력 값에 linear layer와 softmax를 취해 각각의 출력 단어를 만들어낼 수 있는것이다.



이제 각각 위치에 대한 정보를 어떻게 만들어 넣을것인지 이야기해보자.

## Positional Encoding

하나의 문장에 대한 각 단어들의 상대적인 위치정보를 모델에게 알려주기 위해 주기함수를 활용한 공식을 사용한다.

<img src="/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.43.59 PM.png" alt="Screen Shot 2021-06-25 at 5.43.59 PM" style="zoom:50%;" />

PE : positional encoding

pos : 각 단어의 번호		i : 각 단어에 대한 임베딩 값의 위치

![Screen Shot 2021-06-25 at 5.49.07 PM](/Users/sua/Library/Application Support/typora-user-images/Screen Shot 2021-06-25 at 5.49.07 PM.png)

