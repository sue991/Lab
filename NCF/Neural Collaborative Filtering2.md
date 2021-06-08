# Neural Collaborative Filtering2

## 4. EXPERIMENTS

이 섹션에서는 다음 연구 질문에 답할 목적으로 실험을 수행한다.

**RQ1)** 제안된 NCF 방법이 최첨단 implicit collaborative filtering method를 능가하는가?

**RQ2)** 우리가 제안한 optimization framework(log loss with negative sampling)는 권장 작업에서 어떻게 작동하나?

**RQ3)** hidden units의 더 깊은 layers가 user–item interaction data로부터 학습하는 데 더 도움이 되는가?

다음 내용에서는 먼저 실험 설정을 제시한 후 위의 세 가지 연구 질문에 답한다.

### 4.1 Experimental Settings

**Datasets** 공개적으로 액세스할 수 있는 두 개의 데이터 세트를 실험했다. MovieLens와 Pinterest. 두 데이터 세트의 특성은 표 1에 요약되어 있다.

1. **MovieLens.**  이 movie rating dataset는 collaborative filtering algorithms을 평가하는 데 널리 사용되어 왔다. 100만 개의 rating이 포함된 버전을 사용했는데, 각 사용자는 최소 20개의 등급이 있다. explicit feedback 데이터이지만, 우리는  explicit feedback의 implicit signal로부터 학습의 성능을 조사하기 위해 의도적으로 그것을 선택했다. 이를 위해 사용자가 항목의 rating을 지정했는지 여부를 나타내는 각 항목이 0 또는 1로 표시되는 implicit 데이터로 변환했다.
2. **Pinterest.** 이 explicit feedback 데이터는 콘텐츠 기반 이미지 권장 사항을 평가하기 위해 생성된다. 원본 데이터는 매우 크지만 매우 sparse하다. 예를 들어 사용자의 20% 이상이 핀이 하나만 있어 collaborative filtering algorithms을 평가하기가 어렵다. 이와 같이, 우리는 최소 20개의 상호작용(핀)을 가진 사용자만 보존한 MovieLens 데이터와 동일한 방식으로 데이터 세트를 필터링했다. 이로 인해 55,187명의 사용자와 1,500,809개의 interactions가 포함된 데이터의 하위 집합이 생성된다. 각 상호 작용은 사용자가 자신의 보드에 이미지를 고정했는지 여부를 나타낸다.

**Evaluation Protocols.** item 추천의 성능을 평가하기 위해 문헌에서 널리 사용되어 온  leave-one-out  평가를 채택했다.

* leave-one-out : n개의 데이터에서 1개를 Test Set으로 정하고 나머지 n-1개의 데이터로 모델링을 하는 방법을 의미

각 사용자에 대해, 우리는 테스트 세트로 최신 interaction을 유지하고 나머지 데이터를 훈련에 활용했다. 평가할 때  모든 사용자에 대해 모든 item의 순위를 매기는 것은 시간이 너무 많이 걸리기 때문에, 사용자가 interaction하지 않는 100개 항목을 무작위로 샘플링하여 100개 항목 중 테스트 항목을 순위를 매기는 공통 전략을 따랐다. Ranked list의 성능은 Hit Ratio(HR- 적중률)과 Normalized Discounted Cumulative Gain(nDCG)으로 판단한다. 특별한 언급 없이, 우리는 두 matrix에 대한 ranked list을 10으로 잘랐다. 따라서 HR은 테스트 항목이 상위 10위 목록에 있는지 여부를 직관적으로 측정하고 NDCG는 상위 순위에서 더 높은 점수를 할당하여 히트 위치를 설명한다. 각 test user에 대한 두 가지 메트릭스를 계산하고 평균 점수를 기록했다.

* nDCG :  **관련성이 높은** 결과를 **상위권**에 노출시켰는지를 기반으로 하는 평가지표

**Baselines.** 제안된 NCF 방법(GMF, MLP 및 NeuMF)을 다음 방법과 비교했다.

- **ItemPop.** : 아이템은 인기에 따라 interaction 횟수로 순위를 매긴다. 이는 recommendation performance를 벤치마킹하기 위한 비개인화 방법입니다.
- **ItemKNN.** : 이것은 standard item-based collaborative filtering 방법이다. 우리는 [19]의 설정을 따라 implicit 데이터에 맞게 조정했다.
- **BPR.** : 이 방법은 implicit feedback에서 학습하도록 조정된 pairwise ranking loss로 방정식의 MF 모델을 최적화한다. 이것은 아이템 추천을 위한 매우 경쟁적인 기준이다. 우리는 learning rate을 사용하여 이를 변화시키고 최고의 성과를 보고했다.
- **eALS. ** : 이것은 아이템 추천을 위한 최첨단 MF 방법이다. 그것은 방정식의 squared loss 을 최적화하여 관찰되지 않은 모든 interactions를 negative instances로 처리하고 item 인기에 따라 불균일하게 가중치를 부여한다. eALS는 uniform-weighting method WMF보다 우수한 성능을 보이기 때문에 WMF의 성능을 더 이상 보고하지 않는다.

제안된 방법은 사용자와 항목 간의 관계를 모델링하는 것을 목표로 하기 때문에, 우리는 주로 사용자-item 모델과 비교한다.  성능 차이는 (항목-항목 모델인 만큼) 개인화를 위한 사용자 모델에 의해 발생할 수 있기 때문에 SLIM 및 CDAE와 같은 item-item 모델과의 비교는 생략한다.

**Parameter Settings.** 우리는 케라스를 기반으로 제안된 방법을 구현했다. NCF 방법의 하이퍼 파라미터를 결정하기 위해, 우리는 각 사용자에 대해 하나의 interaction을 validation 데이터로 랜덤하게 샘플링하고 그것에 대해 튜닝된 hyper-parameters를 지정했다. 모든 NCF 모델은 방정식 7의 log loss을 최적화함으로써 학습되며, 여기서 positive instance당 네 개의 negative instances를 샘플링했다. 처음부터 훈련된 NCF 모델의 경우 가우스 분포(평균 0, 표준 편차 0.01)로 랜덤하게 모델 매개 변수를 초기화하여 미니 배치 아담으로 모델을 최적화했다. 우리는 [128, 256, 512, 1024]의 batch size와 [0.0001, 0.0005, 0.001, 0.005]의 learning rate를 테스트했다. NCF의 마지막 hidden layer이 capability을 결정하므로, 우리는 그것을 predictive factors로 간주하고 [8, 16, 32, 64]의 factors를 평가했다. 큰 factors은 과적합을 유발하고 성능을 저하시킬 수 있다는 점에 주목할 필요가 있다. 특별한 언급 없이, 우리는 MLP에  세 개의hidden layers를 사용했다. 예를 들어, predictive factors의 크기가 8이라면, NeuCF 레이어의 구조는 32 → 16 → 8이고, 임베딩 크기는 16이다. 사전 훈련된 NeuMF의 경우, α가 0.5로 설정되어 사전 훈련된 GMF와 MLP가 NeuMF의 초기화에 동등하게 기여할 수 있었다.

### 4.2 Performance Comparison (RQ1)

그림 4는 predictive factors의 수와 관련된 HR@10 및 NDCG@10의 성능을 보여준다. MF 방법 BPR 및 eALS의 경우 predictive factors의 수는 latent factors의 수와 같다. ItemKNN의 경우 서로 다른 이웃 크기를 테스트하고 최상의 성능을 보고했다. ItemPop의 성능이 약하기 때문에 개인화된 방법의 성능 차이를 더 잘 강조하기 위해 그림 4에서 생략한다. 

 첫째, NeuMF는 두 데이터 세트에서 최첨단 방법 eALS와 BPR을 크게 능가하여 최고의 성능을 달성한다는 것을 알 수 있다(평균적으로 eALS와 BPR에 비해 상대적 개선은 4.5%, 4.9%임). 핀터레스트의 경우, 예측 인자가 8인 경우에도 NeuMF는 64인 큰 인자로 eALS 및 BPR보다 훨씬 뛰어나다. 이는 선형 MF와 비선형 MLP 모델을 융합하여 NeuMF의 높은 표현성을 나타낸다. 둘째, 다른 두 가지 NCF 방법인 GMF와 MLP도 상당히 강력한 성능을 보여준다. 이 중 MLP는 GMF보다 약간 낮은 성능을 보인다. MLP는 hidden layers을 더 추가하면 더욱 개선될 수 있으며(섹션 4.4 참조), 여기서는 세 layers의 성능만 보여준다. 작은 predictive factors의 경우 GMF는 두 데이터 세트에서 eALs를 능가하는데, GMF는 large factors에 대한 과적합으로 어려움을 겪지만, 얻은 최고의 성능은 eALS의 성능보다 더 우수하다. 마지막으로, GMF와 BPR은 동일한 MF 모델을 학습하지만 objective functions는 다르기 때문에, GMF는 BPR에 비해 일관된 개선을 보여 recommendation task에 대한 classification-aware log loss의 효과를 인정한다. 

 그림 5는 ranking position K가 1부터 10까지인 상위 K recommended lists의 성능을 보여준다. 수치를 보다 명확하게 하기 위해, 우리는 세 가지 NCF 방법 모두보다 NeuMF의 성능을 보여준다. 볼 수 있듯이 NeuMF는 여러 위치에서 다른 방법에 비해 일관된 개선을 보여주며, 1-sampled paired t-검정을 추가로 실시하여 모든 개선이 p < 0.01에 대해 통계적으로 유의하다는 것을 검증하였다. 기준 방법의 경우, eALS는 약 5.1%의 상대적 개선으로 MovieLens의 BPR을 능가하는 반면 NDCG의 관점에서 Pinterest의 BPR을 저하시킨다. 이는 pairwise ranking-aware learner 때문에 BPR이 ranking performance에 강력한 performer가 될 수 있다는 [14]의 연구 결과와 일치한다. neighbor- based  ItemKNN은 모델 기반 방법을 성능이 떨어진다. 그리고 ItemPop은 단순히 사용자에게 인기 있는 아이템을 추천하는 것이 아니라 사용자의 개인화된 선호도를 모델링할 필요성을 나타내며 최악의 성능을 보이고 있다.

### *4.2.1 Utility of Pre-training*

NeuMF에 대한  pre-training의 유용성을 입증하기 위해, 우리는  pre-training을 포함하거나 포함하지 않은 두 가지 버전의 NeuMF의 성능을 비교했다. 사전 훈련이 없는 NeuMF의 경우, 우리는 random initializations를 통해 이를 학습하기 위해 Adam을 사용했다. 표 2에서와 같이, 사전 훈련이 있는 NeuMF는 대부분의 경우 더 나은 성능을 달성한다. predictive factors이 8인 MovieLens의 경우에만 사전 훈련 방법이 약간 더 성능이 떨어진다. pre-training된 NeuMF의 상대적 향상은 MovieLens와 Pinterest가 각각 2.2%, 1.1%이다. 이 결과는 NeuMF 초기화를 위한 사전 훈련 방법의 유용성을 정당화한다.

### 4.3 Log Loss with Negative Sampling (RQ2)

implicit feedback의 단일 클래스 특성을 다루기 위해, 우리는 이진 분류 작업으로 recommendation을 캐스팅했다. NCF를 확률적 모델로 보고 log loss로 최적화했다. 그림 6은 MovieLens에서 각 반복의 NCF 방법의 training loss(모든 인스턴스 평균)과 recommendation 성능을 보여준다. Pinterest에 대한 결과는 동일한 추세를 나타내므로 공간 제한으로 인해 생략됩니다. 첫째, 반복 횟수가 많을수록 NCF 모델의 traing loss가 점차 감소하고 recommendation 성능이 향상됨을 알 수 있다. 가장 효과적인 업데이트는 처음 10회 반복에서 발생하며, 더 많은 반복이 모델에 과적합할 수 있다(예: NeuMF의 training loss가 10회 반복 후에도 계속 감소하지만 recommendation 성능이 실제로 저하된다). 둘째, 세 가지 NCF 방법 중 NeuMF가 가장 낮은 training loss를 달성하고 MLP, 그 다음 GMF를 달성한다. recommendation 성능은 또한 NeuMF > MLP > GMF와 동일한 추세를 보여준다. 위의 연구 결과는 implicit 데이터에서 학습하기 위해 log loss을 최적화하는 합리성과 효과에 대한 경험적 증거를 제공한다. pairwise objective functions에 비해 pointwise log loss의 장점은 negative instances에 대한 유연한 샘플링 비율이다. pairwise objective functions는 하나의 샘플링된 음의 인스턴스만 양의 인스턴스와 쌍으로 구성할 수 있지만, 우리는 pointwise loss의 샘플링 비율을 유연하게 제어할 수 있다. NCF 방법에 대한 음성 샘플링의 영향을 설명하기 위해 그림 7에서 서로 다른 negative 샘플링 비율에 대한 NCF 방법의 성능을 보여준다. 양성 인스턴스당 단 하나의 음수 샘플이 최적의 성능을 달성하기에 불충분하며, 음수 인스턴스를 더 많이 샘플링하는 것이 유익하다는 것을 명확히 알 수 있다. GMF와 BPR을 비교해보면, 우리는 1의 샘플링 비율을 가진 GMF의 성능이 BPR과 동등하다는 것을 알 수 있고, GMF는 더 큰 샘플링 비율을 가진 BPR이 훨씬 더 우수하다는 것을 알 수 있다. 이는 pairwise BPR loss보다 pointwise log loss의 이점을 보여준다. 두 데이터 세트의 경우, 최적의 샘플링 비율은 약 3에서 6이다. 핀터레스트에서 표본 추출 비율이 7보다 크면 NCF 방법의 성능이 떨어지기 시작한다는 것을 발견했다. 이는 샘플링 비율을 지나치게 높게 설정하면 성능에 악영향을 미칠 수 있음을 나타낸다.

### 4.4 Is Deep Learning Helpful? (RQ3)

neural networks과 user–item interaction 함수를 학습하는 작업은 거의 없기 때문에 deep network structure를 사용하는 것이 recommendation task에 도움이 되는지 궁금하다. 이를 위해 hidden layer수가 다른 MLP를 추가로 조사했다. 그 결과는 표 3과 4에 요약되어 있다. MLP-3은 (em- bedding layer 외에) 세 개의 hidden layers가 있는 MLP method를 나타내고 다른것도 비슷하다. 우리가 볼 수 있듯이, 동일한 기능을 가진 모델의 경우에도 더 많은 레이어를 쌓는 것이 성능에 유익하다. 이 결과는 매우 고무적인 것으로, collaborative recommendation을 위해 deep models을 사용하는 것의 효과를 나타낸다. 우리는 이러한 개선이 더 많은 non-linear layers를 쌓음으로써 초래된 높은 비선형성 때문이라고 본다. 이를 확인하기 위해, 우리는 identity function를 activation 함수로 사용하여 linear layers를 쌓아 보았다. 성능은 ReLU 유닛을 사용하는 것보다 훨씬 더 나쁘다. 

hidden layers가 없는 MLP-0의 경우(즉, 임베딩 계층이 예측에 직접 투영됨) 성능은 매우 약하며 비개인화된 ItemPop보다 낫지 않다. 이는 3.3 섹션의 우리의 주장을 입증하는 것으로 단순히 사용자와 항목의 latent  vector를 연결하는 것이 feature interactions을 모델링하기에 불충분하므로 hidden layers으로 변환할 필요가 있다는 것이다.

## 5. RELATED WORK

권고사항에 대한 초기 문헌은 주로 explicit 피드백에 초점을 맞추고 있지만, 최근의 관심은 점점 더 implicit 데이터로 옮겨가고 있다. implicit 피드백이 있는 collaborative filtering(CF) 작업은 일반적으로 item recommendation 문제로 공식화되며, 그 목적은 사용자에게 짧은 items a리스트를 권장하는 것이다. explicit 피드백에 대한 작업으로 광범위하게 해결된rating prediction과는 대조적으로, item recommendation problem를 해결하는 것은 더 실용적이지만 어려운 일이다. 한 가지 핵심 통찰력은 누락된 데이터를 모델링하는 것인데, 이는 explicit 피드백에 대한 작업에서 항상 무시됩니다. implicit 피드백으로  item recommendation에 대한 latent factor 모델을 맞춤화하기 위해, 초기 연구[19, 27]는 두 가지 전략이 제안된 곳에서 균일한 가중치를 적용한다. 즉, 모든 결측 데이터를 negative instances 또는 결측 데이터에서 샘플링된 negative instances로 처리했다. 최근에, He ..와 Liang ..들이 weight missing data에 대한 전용 모델을 제안했고, Rendle 등이 제안하였다.  feature-based factorization 모델을 위한 implicit coordinate descent(iCD) 솔루션을 개발하여 item recommendation을 위한 최첨단 성능을 달성하였다. 다음에서는 neural networks을 사용하는 recommendation work에 대해 논의한다. Salakhutdinov 등 초기 개척자들의 연구는 item에 대한 사용자의 explicit ratings를 모델링하기 위해 two-layer Restricted Boltzmann Machines(RBMs)를 제안했다. 이 작업은 나중에 등급의 서수적 성격을 본보기로 확장되었다. 최근, autoencoders는 recommendation systems을 만들기 위한 인기 있는 선택이 되었다. 사용자 기반 AutoRec의 아이디어는 과거 ratings을 입력으로 간주하여 사용자의 rating을 재구성할 수 있는 hidden structures를 학습하는 것이다. 사용자 개인화의 관점에서, 이 접근법은 사용자를 등급 항목으로 나타내는 항목-항목 모델과 유사한 정신을 공유한다. autoencoder가 identity function를 학습하고 보이지 않는 데이터로 일반화하지 못하는 것을 방지하기 위해 의도적으로 손상된 입력에서 학습하기 위해 denoising autoencoders(DAE)가 적용되었다. 좀 더 최근에, Zheng 등은 CF에 대한 neural autoregressive method을 제시하였다. 이전의 노력이 CF를 해결하기 위한 신경망의 효과에 대한 지원을 제공했지만, 대부분은 explicit ratings에 초점을 맞추고 관찰된 데이터만 모델링했다. 결과적으로, 그들은 positive-only implicit data에서 사용자의 선호도를 쉽게 배우지 못할 수 있다. 최근 일부 연구는 implicit 피드백을 기반으로 한 추천을 위한 딥 러닝 모델을 탐구했지만, 그들은 주로 항목에 대한 텍스트 설명, 음악의 음향 특징, 사용자의 교차 도메인 행동 및 지식 기반의 풍부한 정보와 같은 보조 정보를 모델링하기 위해 DNN을 사용했다. 그런 다음 DNN에서 학습한 features가 CF의 MF와 통합되었다. 우리의 작업과 가장 관련이 있는 작업은 [44]이며, implicit 피드백이 있는 CF를 위한 collaborative denoising autoencoder(CDAE)를 제시한다. DAE 기반 CF와 대조적으로, CDAE는 사용자 ratings를 재구성하기 위해 autoencoder의 입력에 사용자 노드를 추가로 연결한다. 저자들이 보여주듯이, identity 함수가 CDAE의 숨겨진 레이어 f를 활성화하기 위해 적용될 때, CDAE는 SVD++ 모델과 동등하다. 이는 CDAE가 CF를 위한 neural modelling 접근 방식이지만, 사용자-항목 상호 작용을 모델링하기 위해 linear kernel(즉, inner product)을 여전히 적용한다는 것을 암시한다. 이는 부분적으로 CDAE에 딥 레이어를 사용해도 성능이 향상되지 않는 이유를 설명할 수 있다. CDAE와는 달리, 우리의 NCF는 multi- layer feedforward neural network와 사용자-항목 상호 작용을 모델링하는wo-pathway architecture를 채택한다. 이를 통해 NCF는 데이터에서 임의의 함수를 학습하여 고정된 inner product function보다 더 강력하고 표현력이 뛰어나다. 유사한 노선을 따라, 두 entities의 관계를 학습하는 것은 지식 그래프의 문헌에서 집중적으로 연구되었다. 많은 관계형 기계 학습 방법이 고안되었다. 우리의 제안과 가장 유사한 것은 neural networks을 사용하여 두 entities의 interaction을 학습하고 강력한 성능을 보여주는 Neural Tensor Network(NTN)이다. 여기서는 CF의 다른 문제 설정에 초점을 맞춘다. MF와 MLP를 결합한 NeuMF의 아이디어는 부분적으로 NTN에서 영감을 얻었지만, 우리의 NeuMF는 MF와 MLP가 서로 다른 임베딩 세트를 학습할 수 있다는 점에서 NTN보다 더 유연하고 일반적이다. 더 최근에는 구글이 앱 추천을 위한 Wide & Deep learning접근 방식을 공개했다. Deep component는 유사한 방식으로 feature embedding에 MLP를 사용하며, 이는 강력한 일반화 능력을 가진 것으로 보고되었다. 이들의 연구는 사용자와 항목의 다양한 기능을 통합하는 데 초점을 맞췄지만, 우리는 순수한 collaborative filtering systems을 위한 DNN을 탐색하는 것을 목표로 한다. 우리는 DNN이 사용자-항목 상호 작용을 모델링하기 위한 유망한 선택임을 보여주며, 우리가 아는 한 이전에는 조사되지 않았다.

## 6. CONCLUSION AND FUTURE WORK

본 연구에서는 협업 필터링을 위한 신경망 아키텍처를 탐구하였다. 우리는 일반적인 프레임워크 NCF를 고안하고 사용자-항목 상호 작용을 다른 방식으로 모델링하는 세 가지 인스턴스(GMF, MLP 및 NeuMF)를 제안했다. 우리의 프레임워크는 단순하고 일반적이다. 이것은 본 논문에서 제시된 모델에 국한되지 않고, recommendation을 위한 딥 러닝 방법을 개발하기 위한 지침으로 설계되었다. 이 연구는 collaborative filtering을 위한 주된 shallow model을 보완하여 딥 러닝을 기반으로 한 추천을 위한 새로운 연구 가능성을 열어준다. 앞으로 NCF 모델에 대한 pairwise learners를 연구하고 사용자 리뷰, 기술 자료 및 시간 신호와 같은 보조 정보를 모델링하기 위해 NCF를 확장할 것이다. 기존 개인화 모델은 주로 개인에 초점을 맞춰왔지만, 소셜 그룹의 의사결정에 도움이 되는 사용자 그룹을 위한 모델을 개발하는 것은 흥미롭다. 또한, 우리는 특히 흥미로운 작업인 멀티미디어 항목에 대한 추천자 시스템을 구축하는 데 관심이 있지만 추천 커뮤니티에서 상대적으로 덜 정밀한 조사를 받았다. 이미지와 비디오와 같은 멀티미디어 항목은 사용자의 관심을 반영할 수 있는 훨씬 풍부한 시각적 의미론을 포함하고 있다. 멀티 미디어 추천 시스템을 구축하려면 멀티 뷰 및 멀티 모달 데이터에서 학습할 수 있는 효과적인 방법을 개발해야 한다. 또 다른 새로운 방향은 효율적인 online recommendatio을 제공하기 위한 ecurrent neural networks과 hashing 방법의 가능성을 탐색하는 것이다.

## 7.Acknowledgement

저자들은 추천 시스템과 논문 개정에 대한 저자들의 생각에 도움이 되는 귀중한 논평에 대해 익명의 검토자들에게 감사한다.