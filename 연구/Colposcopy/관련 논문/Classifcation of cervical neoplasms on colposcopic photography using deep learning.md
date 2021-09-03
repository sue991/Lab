# Classifcation of cervical neoplasms on colposcopic photography using deep learning(2020)

## Abstract

Coploscopy는 cervical cancers를 감지하는데 널리 사용되지만, 전문의가 부족하다.

따라서 AI를 이용하는데, pre-trained된 CNN이 두가지 grading system에 맞게 fine-funed된다:

**cervical intraepithelial neoplasia (CIN)system** 과 **lower anogenital squamous terminology (LAST) system** 이다.

CIN에 대한 **multi‐class classification** 정확도는 **Inception‐Resnet‐v2** 에서 **48.6 ± 1.3%** 이고, **Resnet‐152**에서 **51.7 ± 5.2%**이다.

LAST에서는 각각 **71.8 ± 1.8%,  74.7 ± 1.8%** 이다.

조직검사가 필요한 병변도 효율적으로 검출되었으며(AUC, 0.947 ± 0.030 by Resnet2152), 주의 지도에 유의미하게 표시되었다.

## Introduction

Cervical cancer는 전 세계 여성들에게서 네 번째로 흔한 암이고 개발도상국 여성들에게서 두 번째로 흔한 암이다.

검사가 중요하지만, 개발도상국에서는 전문의가 흔치않다.

이전 연구에서는 colposcopic 진단 및 동반 생체검사의 정확도를 평가하여 민감도가 70.9~98%이고 특이성이 45~90%이다.

그러나 colposcopic 진단의 정확도는 의사의 기술에 따라 크게 달라져 제공업체 간의 민감도 및 특수성이 크게 달라진다.

이로 인해 경부암 누락 등 병변 과소진단이나 병변 과대진단 등이 우려되면서 저급 경부병변 과대치료, 감염 위험 증가, 환자 불편, 재정 부담 등이 우려되고 있다.

  따라서 CNN 을 이용한 AI가 많이 도입되고 있는데, 두가지에 중점을 두고 있다.

주된 목적은 machine learning-based colposcopy model을 개발하는 것인데, 이 모델은 CIN system과 LAST system의 두 histopathologic systems을 사용하여 cervical neoplasms을 자동으로 분류한다.

두번째 목적은 생체검사를 필요로 하는 cervical lesions을 확인하는 Ai model의 성능을 평가하는 것이다.

우리가 아는 바로는, 이것은 조직검사에서 확인된 전암 환자 중 colposcopic photograph 판독에 인공지능을 적용하는 것에 대한 가장 큰 연구 중 하나이다.

## Materials and methods

  ### **Study subjects**

모든 neoplastic lesions은 인간 유두종바이러스(HPV) 검사를 받았다(그림 1A 및 표 1).

사진을 찍은 후, 미국 대장내시경 및 자궁경부병리학회 4의 지침에 따라 6000건이 넘는 경험을 가진 전문 산부인과 전문의들이 대장내시경 지시 생검과 절개를 수행했다.

모든 사진은 수술 또는 침습적 시술 전에 대장경 검사 중에 촬영되었다.

정상 식염수를 도포한 후 3–5% 아세트산으로 세척한 후 변환 구역과 관심 부위를 평가했다.

본 연구에서는 백색광 이미지만 사용되었으며, 640 × 480 픽셀 해상도의 참여 병원 사진 보관 및 통신 시스템에서 검색되었다. 화질이 불량하거나 초점이 맞지 않는 영상과 같이 적절한 분류를 허용하지 않는 영상은 연구에서 제외되었으며, 모든 개인 식별자가 제거되었다.

### Classification of cervical lesions

모아진 이미지는 두개의 histopathologic systems을 classify 하는데 사용되었다.

각 시스템에 대해 경부 병변을 고위험 대 저위험 병변으로 이분화하는 이진 분류 모델을 설계했다.

자궁경부 병변에 대한 조직검사의 **필요성을 결정**하기 위해  binary decision model이 개발되었다.

따라서 Need-To-Biopsy는 '정상적이지 않음'으로 정의되었다. 따라서 Need-To-Biopsy 시스템에는 Normal과 Need-To-Biopsy라는 두 가지 클래스만 있다(그림 1B).

### **Construction of datasets**

데이터셋은 training과 test dataset으로 나뉜다(85%, 15% 비율).

나눈 후, training dataset에 data augmentation을 수행하여 class imbalance를 줄인다.

Model의 robustness를 확인하기 위한 under-sampling을 위해 세 가지 다른 seed numbers를 사용하여 3가지 training dataset 조합을 만들었다. 마지막으로, training set은 val set을 만들기 위해 75:10 비율로 나뉘었다.

### Pre-processing of datasets

모든 이미지는 원래 해상도인 640 × 480 픽셀에서 새로운 해상도인 480 × 480 픽셀로 자동 중앙 자르기 과정을 거쳤으며, 각 오른쪽과 왼쪽 여백에 대해 80 픽셀을 제거했다.

모든 이미지에 대해 min-max normalization을 적용하였다.

교육 데이터셋의 회전 이미지를 추가하여 각 모델에 대해 구축된 각 교육 데이터셋에 대해 data augmentation을 맞춤화했다.

CIN system을 위한 multi-class classification model에서, cancer group은 원본 이미지를 30°, 60°, 90°, 120°, 150° 회전시켜 6배 증강되었다.

### **Training of the CNN models.**

두 개의 CNN model이 적용되었고, 각각 Inception-Resnet-v2 model과  Resnet-152이다.

CNN 모델은 ImageNet 가중치에 의해 사전 교육되었으며 본 연구의 colposcopy를 사용하여 미세 조정되었다.

우리는 단순히 출력을 이진 클래스로 변환하여 다중 클래스 분류기에서 결과를 유도하지 않았다.

Categorical cross-entropy는 다중 클래스 분류에서 손실 함수로 사용되었고, binary cross-entropy는 이진 분류에 사용되었다. 모든 교육은 PyTorch 플랫폼을 사용하여 수행되었다. 

  모델 교육은 세 단계로 구성되었는데, 첫 번째 단계에서는 해상도가 400 × 400, 두 번째 단계에서는 450 × 450 해상도, 마지막 단계에서는 480 × 480 해상도의 이미지가 감소했다.

(1) pre-trained model을 로딩하고, 마지막 layers만 unfreezing한 다음, training을 수행한다.

(2) 전체 layers를 unfreezing하고 처음 몇개, 중간 및 마지막 layer에 대해 다른 learning rates를 적용하여 cyclically하게 training 한다.

각 단계에서, cyclic learning rate schedule을 수행했으나, snapshot ensemble은 적용하지 않았다.

요약하자면, 초기 learning rate는 1e-3으로 선택되었으며, 이는 교육을 시작하기 전에 단일 학습 속도 범위 테스트에서 가장 낮은 검증 손실을 나타냈다. 

그런 다음 learning rate는 한 사이클 내에 cosine annealing을 수행한 후 다음 사이클이 시작될 때 초기 학습 속도로 되돌아갔다.

각 단계에서 길이가 1, 4, 16, 64인 4개의 사이클이 사용되었다. 각 사이클에서 validation loss를 최소화하기 위해 early stopping을 사용했다.

2단계에서는 서로 다른 하위 계층에 대해 세 가지 차등 학습률이 사용되었으며, 초기 학습률은 (1e-3)/9, (1e-3)/6 및 1e-3이었다. dropout 비율은 0.5로 구현되었다.

### **Class activation map (CAM).**

각 CNN 아키텍처에 대해 마지막 몇 개의 레이어는 컨볼루션 레이어를 추가하기 전에 제거되었으며 global average pooling 및 softmax layers가 적용되었다.

클래스를 결정하는 데 중요한 크기를 나타내기 위해 각 해당 클래스 피쳐 가중치에 대해 전역 평균 풀링을 사용하여 공간적으로 풀링된 피쳐 맵을 곱했다.

업샘플링은 원본 이미지의 로컬라이제이션으로 리디렉션된다. 이 방법을 사용하여 각 결과에 대한 클래스 활성화 맵이 제시되었다. 우리는 가장 활성화된 지역을 나타내기 위해 빨간색을 선택했다.

### **Main outcome measures and statistical analysis.**

테스트 데이터셋의 유리 예측은 원본, 수평 플립, 수직 플립, 수평 플립, 수평 플립 이미지를 포함한 4가지 증강 기능을 사용하여 테스트 시간 확대(TTA)를 통해 수행되었다. TTA의 목표는 다양한 관점에서 이미지를 사용하여 예측 정확도를 높이는 것이었다. 단일 이미지에 대해 네 가지 예측을 수행했으며, 네 가지 예측의 평균을 최종 예측으로 삼았다.

모델 성능을 평가하기 위해 시드 번호가 서로 다른 세 가지 교육 데이터 세트가 사용되었다.



