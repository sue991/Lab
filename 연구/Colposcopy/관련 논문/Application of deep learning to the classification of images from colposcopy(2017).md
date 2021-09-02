# Application of deep learning to the classification of images from colposcopy(2017)

## Abstract

Deep learning을 colposcopy image classification에 잘 적용하고 싶다.

이를 위해 총 158명의 인공수정을 받은 환자가 등록되었으며, 산부인과 종양학 데이터베이스의 의료 기록과 데이터를 소급 검토하였다.

colposcopy의 수술 전 데이터가 input data로 사용되었으며, [severe dysplasia, carcinoma *in situ* (CIS) and invasive cancer (IC)]. 세 가지로 분류된다.

485개의 이미지가 analysis에 사용됐는데, 그 중 142개는  severe dysplasia(심각한 형성) (2.9 images/patient), 257개는 CIS(3.3 images/patient), 86개는 IC (4.1 images/patient)이다.

여기서 233개는 green filter를, 나머지 252개는 green filter 없이 캡처되었다.

L2 정규화, L1 정규화, 드롭아웃 및 data augmentation을 적용한 후  validation dataset의 정확도는 ~ 50%였다.

본 연구는 예비 연구이지만, 결과는 딥러닝을 적용하여 대장내시경 영상을 분류할 수 있다고 나타났다.

## Introduction

  자궁경부암은 전 세계 여성의 주요 사망 원인이다.

비록 Pap smear test의 도입으로 사망률이 급격히 감소했지만, 특히 약물 과다 복용을 피하기 위해, 더 이상 검사되고 고위험으로 취급되어야 하는 환자의 유형을 결정하는 것은 여전히 중요한 문제로 남아 있다.

일상에서 환자 관리는 cytology(세포학), histology(조직학), HPV typing 및 colposcopy 결과를 함께 사용하여 결정된다.

본 연구에서는 colposcopy image에 초점을 맞춘 딥러닝이 수술 후 진단을 예측할 수 있는지 조사했다.

  자궁경부 상피내 병변은 아세트산 용액으로 치료하면 쉽게 인식된다.
예를 들어, 아세트산(acetic acid) 처리 후 하얗게 변하는 부위(아세토 화이트닝) 및/또는 비정상적인 혈관 패턴을 보이는 부위가 조직검사에서 고려된다. 이러한 효과는 녹색 필터를 적용한 후 더 잘 보인다.

그런 다음 산부인과 의사가 얼룩의 정도와 기본 혈관 패턴을 기준으로 진단을 평가한다.

Data input 중에 이 도트 패턴의 존재와 같은 이미지의 특징이나 표현이 선택되지 않았기 때문에 현재 연구는 앞서 언급한 스터디와 구별된다.

  또한 본 연구에서는 획득한 image의 수가 충분하지 않기 때문에 학습 속도를 개선하고 과적합을 방지하기 위한 방법을 조사했다.

제시된 과정에서는 L2정규화, L1정규화, 드롭아웃이 적용되었으며, 데이터 증대를 통해 입력 데이터의 양이 증가하였다.

본 연구에서는 정확도 자체를 강조하는 것이 아니라 인공지능이나 기계학습 전문가가 아닌 산부인과 의사가 임상 실무에서 딥러닝을 활용할 수 있다는 것을 보여주려는 의도이다.

또한, 현재 결과는 임상 실무의 관련 정보를 향후 사용을 위해 적절히 저장해야 한다는 것을 시사한다.

## Materials and Methods

*Patients*.  각 진단은 수술 후 병리(consization)에 따라 원칙적으로 수행되었지만, 결과가 심할 때는 수술 전 병리(biopsy)를 우선시하여 대장내시경 영상의 출력(‘target’ in deep learning)으로 사용하였다.

158명의 환자가 등록되었고, 평균 나이는 39세이다.(21-63세)

진단 및 해당하는 환자 수는 다음과 같다: severe dysplasia, 49, carcinoma *in situ* (CIS), 78, invasive cancer(IC), 21, 그리고 다른 것들(such as adenocarcinoma *in situ* and invasive adenocarcinoma), 10.

현재 스터디에서는 사용 가능한 영상 수가 제한되어 있어 환자 분류가 세 그룹(심각한 이형성, CIS 및 IC)으로 제한되었다.

*Images*.   colposcopy 수술 전 영상이 딥러닝을 위한 입력 데이터로 사용되었다. 이 조사는 소급 연구였기 때문에 보관할 colposcopy images의 수와 유형을 결정하는 기준은 없었다.

생검 부위를 나타내며 진단에 사용된 녹색 필터를 사용하거나 사용하지 않은 아세트산 처리 후 영상이 저장되었다.

485개의 이미지 중 142개는 severe dysplasia (2.9 images/patient), 257개의 CIS (3.3 images/patient), 86개 IC (4.1 images/patient) 이미지 이다. 이것 중, 233개는  green filter, 252개는 green filter가 없는 이미지 이다.

Image는 640x480x3 pixels로 저장되었다. 이 이미지들은 300x300으로 다듬어졌다.

이 이미지는 딥러닝에서 150x150으로 또 다듬어진다.

*Deep learning.* [Code]([https://blog.keras.io/building‐powerful‐image‐classificat ion‐models‐using‐very‐little‐data.html])는 여기에 있다. 

## Results

*Images*  통계 분석 결과, 더 심각한 병변에는 더 많은 수의 영상이 저장되는 것으로 나타났다(P=0.0085). 전체 이미지 수는 녹색 필터의 유무에 따라 입력 이미지를 나누는 것보다 과대 맞춤을 방지하는 데 더 중요하다.

수집된 이미지 세트에는 녹색 필터가 있는 이미지와 없는 이미지가 모두 포함되었으며, 혼합 데이터로 인한 학습 비효율성으로 인해 이러한 이미지가 개별적으로 학습에 사용되었다.

그러나 유효성 검사 데이터 세트가 다시 선택되었으며(각 진단당 10개 영상, 총 30개), 결과는 녹색 필터의 유무에 관계없이 유효성 검사 정확도가 감소했음을 입증했다.

이 결과는 총 영상 수 감소와 관련이 있을 수 있다.

따라서, 적어도 현재의 소규모 연구에서 총 이미지 수는 녹색 필터의 유무에 따른 입력 데이터의 분할보다 유효성 검사 정확도를 높이는 데 더 중요한 것으로 나타났다.

*L2 regularization can improve overfitting.*  과부착을 방지하기 위해 L2 정규화, L1 정규화 및 드롭아웃을 적용했다. 첫 번째 입력 계층에는 L2 정규화 및 L1 정규화가 적용되었으며, 최대 풀링 후 모든 계층에 드롭아웃이 적용되었다(드롭아웃 속도는 0.5로 설정됨).

L2 정규화는 적절하게 튜닝했을 때 과부착을 방지하는 데 효과적인 것으로 나타났다.

*Data augmentation slightly improves the validation accuracy and overfitting.*

단일 이미지에서 20개의 이미지를 무작위로 회전하고, 확대/축소 배율을 적용하고, 이미지를 수평으로 플립한 다음 결과 이미지를 입력으로 사용했다.

데이터 확대에 의해 하나의 영상이 최대 20개의 영상으로 변환되었다.

