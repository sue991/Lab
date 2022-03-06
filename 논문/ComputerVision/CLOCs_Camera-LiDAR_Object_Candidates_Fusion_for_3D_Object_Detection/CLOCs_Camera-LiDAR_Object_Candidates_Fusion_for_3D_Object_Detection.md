# MOTIVATION

## 2D and 3D Object Detection

### 2D detection systems 

input -  RGB image

output - 2D axis-aligned bounding boxes with confidence scores

### 3D detection systems

generate - classified oriented 3D bounding boxes with confidence scores



카메라와 LiDAR의 calibration parameters를 사용하여 LiDAR 좌표의 3D 경계 상자를  image plane에 정확하게 투영할 수 있다.

## Why Fusion of Detection Candidates

퓨전 아키텍처는 여러 모달의 처리 기능이 결합되는 시점에 따라 분류될 수 있다.

세 가지 일반적인 범주는 (1) 입력의 데이터를 결합하는 초기 융합, (2) 중간 특징을 동시에 결합하면서 서로 다른 모달리티에 대해 서로 다른 네트워크를 갖는 심층 융합, (3) 개별 경로에서 각 모달리티를 처리하고 의사 결정 수준의 출력을 융합하는 후기 융합이다.

**초기 융합**은 교차 모달 상호 작용을 위한 가장 큰 기회를 가지고 있지만, 동시에 정렬, 표현 및 희소성을 포함한 양식 간의 고유한 데이터 차이가 동일한 네트워크를 통해 전달된다고 해서 반드시 잘 설명되는 것은 아니다.

**심층 융합**은 처리 중에도 여러 가지 기능을 결합하면서 서로 다른 모달리티에 대해 별도의 채널을 포함함으로써 이 문제를 해결한다.

이것은 가장 복잡한 접근 방식이며, 실제로 복잡성이 실제 개선으로 이어지는지를 확인하는 것은 쉽지 않다. 즉 단순히 single-modality methods에 대한 이득을 보여주는 것만으로는 충분하지 않다.

**후기 융합**은 training에 상당한 이점이 있다. single modality algorithms은 자체 센서 데이터를 사용하여 학습할 수 있다. 따라서 multi-modal data를 다른 modalities와 동기화하거나 정렬할 필요가 없다.

오직 후기 융합 단계만이 공동으로 정렬하고 레이블링된 데이터를 필요로 한다. 또한 후기 융합이 작동하는 detection candidate data는 작고 네트워크를 인코딩하기 쉽다.

후기 융합은 새로운 detection을 만들기 보다는 pruning을 하기 때문에, input detector를 precision이 아닌 recall을 최대화하도록 조정하는 것이 중요하다. 실제로 이는 개별 modalities (a)가 실제 탐지를 실수로 억제할 수 있는 NMS 단계를 피한다는 것을 의미한다. 그리고 (b) 임계값을 가능한 낮게 유지한다.

우리의 late fusion framework에서, NMS 이전에 모든 detection candidates를 fusion step에서 통합하여 모든 잠재적 correct detection 추출 확률을 극대화한다.

우리의 접근 방식은 데이터 기반이다 ; 우리는 detection candidates의 spatial description 뿐만 아니라 individual detection candidates의 output scores와 classification을 input으로 받는 discriminative network를 train한다. 이는 데이터로부터 최종 output detection을 위해 input detection candidates를 결합하는 가장 좋은 방법을 배운다.

# CAMERA-LIDAR OBJECT CANDIDATES FUSION

## *Geometric and Semantic Consistencies*

주어진 이미지 프레임과 LiDAR 데이터의 경우, 단일 3D 탐지 및 점수 세트를 찾는 각 양식에는 다양한 신뢰도를 가진 많은 탐지 후보가 있을 수 있다. 이러한 탐지 후보를 결합하려면 고유한 연결이 아니더라도 서로 다른 양식 간의 연결이 필요하다. 이를 위해 기하학적 연관성 점수를 작성하고 의미적 일관성을 적용한다. 

### Geometric consistency

2D 및 3D 디텍터에서 올바르게 감지된 개체는 이미지 평면에서 동일한 경계 상자를 가진다. 반면 false positives 경계 상자는 동일한 경계 상자를 가질 가능성이 낮다. 포즈의 작은 오류가 발생하면 겹침이 줄어든다. 이는 2D 경계 상자의 이미지 기반 교차점(IoU)과 3D 탐지의 투영 모서리의 경계 상자를 동기 부여하여 2D와 3D 탐지의 기하학적 일관성을 정량화한다.

### Semantic consistency

디텍터는 여러 범주의 개체를 출력할 수 있지만 퓨전 중 동일한 범주의 탐지만을 연결한다. 이 단계에서 임계값 탐지를 피하고(또는 매우 낮은 임계값을 사용하고) 최종 퓨전 점수를 기준으로 임계값을 최종 출력에 둔다.

위에서 설명한 두 가지 유형의 consistency가 퓨전 네트워크에서 사용되는 기본 개념이다.

## *Network Architecture*

융합된 데이터의 preprocessing/encoding, fusion network architecture 및 교육에 사용되는 손실 함수에 대해 설명한다.

###  *Sparse Input Tensor Representation:*

인코딩 단계의 목표는 모든 individual 2D 및 3D 탐지 후보를 퓨전 네트워크에 공급할 수 있는 일관된 joint detection candidates 세트로 변환하는 것이다.

2D object 디텍터의 일반적인 output은  image plane의 2D 경계 상자 세트와 그에 상응하는 confident scores이다. 한 이미지에서 k개의 2D detection candidates는 다음과 같이 정의할 수 있다.
$$
P^{2D} ={p^{2D}_1, p^{2D}_2, ... ,p^{2D}_k}, \\
p^{2D}_i = \{  [x_{i1},y_{i1},x _{i2},y_{i2} ],s^{2D}_i \}
$$
P_2D는 한 image에서 k개의 detection candidates의 집합이고, p_2D는 2D bounding box로부터 xi1,yi1의 top left, xi2,yi2의 bottom right 의 pixel 좌표값을 가진 i번째 detection이다. s_2D는 confident score 이다.

3D object detectors의 output은 LiDAR 좌표의 3D oriented bounding boxes와 confident scores이다.

3D bounding boxes를 인코딩하는데 여러 방법이 있는데, 3D 치수(높이, 너비 및 길이), 3D 위치(x,y,z), rotation (yaw angle)를 포함하는 7자리 벡터가 사용된다. 

하나의 LiDAR scan에 n개의 3D detection candidates를 다음과 같이 정의한다 :
$$
P^{3D} = {p^{3D}_1, p^{3D}_2, ...,p^{3D}_n}, \\
p^{3D}_i =\{ [h_i ,w_i ,l_i ,x_i ,y_i ,z_i ,θ_i ], s^{3D}_i \}
$$
 P_3D : set of all n detection candidates in one LiDAR scan

p_3D : 3D bounding box의 7자리 vector. s_3D  : 3D confident score

NMS를 수행하지 않고 2D 및 3D 탐지를 수행한다는 점에 유의하자. 이전 절에서 논의한 것처럼 일부 올바른 탐지는 single sensor modality의 제한된 정보 때문에 억제될 수 있다.  우리가 제안한 융합 네트워크는 더 나은 예측을 위해 두 센서 모달리티 모두에서 모든 탐지 후보를 재평가할 것이다.

k 2D detections와 n 3D detections에서, `k × n × 4` tensor T 를 만든다. 각 요소 Ti,j에는 다음과 같은 4개의 채널이 있다 :
$$
T_{i,j} = \{ IoU_{i,j} , s^{2D}_i,s^{3D}_j, d_j \}
$$
IoU_ij : i번째 2D Detection과 j번째 projected 3D detection의 IoU

s2D, s3D : 각각 ith 2D와 jth 3D detection의 confident score

dj : xy plane에서 j-th 3D bounding box와 LiDAR사이의 normalized distance

IoU가 0인 Ti,j element는 기하학적으로 일관성이 없기 때문에 제거된다.

input tensor T는 각 투영된 3D detection에 대해 몇 개의 2D detection만 교차하므로 대부분의 elements가 비어 있기 때문에 sparse하다.

Fusion network는 intersected examples로부터만 학습하면된다.

텐서 T의 희소성을 활용하고 큰 k 및 n 값에 대해 훨씬 빠르고 실현 가능한 계산을 수행하기 위한 구현 아키텍처를 제안한다.

비어 있지 않은 요소만 processing을 위해 융합 네트워크에 전달된다.

여기서 2D 검출이 없는 투영된 3D 검출 p의 경우 여전히 J번째 열 Tk,j의 마지막 요소를 사용 가능한 3D 검출 정보로 채우고 IoUk,j, s2D k를 -1로 설정한다.

##  *Network Details*

fusion network 는  1×1 2D convolution layers의 집합이다.

Conv2D(cin, cout, k, s)

4개의 conv layer 순차적으로 사용

`Conv2D(4, 18, (1,1), 1), Conv2D(18, 36, (1,1), 1), Conv2D(36, 36, (1,1), 1) and Conv2D(36, 1, (1,1), 1)`

1 × p × 1 size tensor를 만듦. p :  the number of non-empty elements in the input tensor T.

non-empty elements (i,j)를 저장해두었기 때문에, k × n × 1 모양의 T_out을 만들 수 있다.

### loss :  cross entropy loss

### *Training* : SGD (lr : 3 * 10−3)
