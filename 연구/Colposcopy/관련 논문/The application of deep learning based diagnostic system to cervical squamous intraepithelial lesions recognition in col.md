# The application of deep learning based diagnostic system to cervical squamous intraepithelial lesions recognition in colposcopy images

***Background***

  자궁경부암 검진의 무거운 부담을 덜어주기 위해서는 과학적이고 정확하며 효율적인 진단 및 조직검사 보조방법 마련이 시급하다.

***Methods*** 데이터는 세 가지 딥러닝 기반 모델을 구축하기 위해 수집되었다.

모든 경우에 대해 **saline image** 1개, **acetic image** 1개, **iodine image** 1개 및 연령,  **human papillomavirus testing** 및 **cytology**,**type of transformation zone**,  **pathologic diagnosis** 등 해당 임상 정보가 수집되었다.

Train, test, val = 8 : 1 : 1 로 나뉘었다.

모델 설정 후, 모델 성능을 추가로 평가하기 위해 고화질 이미지의 독립 데이터 세트가 수집되었다.

또한 전문의와 모델 간의 진단 정확도 비교도 수행했다.

***Results*** 음성 환자를 양성 환자와 구별하기 위한 분류 모형의 민감도, 특이성 및 정확도는 각각 85.38%, 82.62%, 84.10%였으며 AUC는 0.93이었다.

초토화 영상에서 의심스러운 병변을 분할하기 위한 분할 모델의 리콜 및 DICE는 84.73%와 61.64%였으며 평균 정확도는 95.59%였습니다. 또한, 84.67%의 고급 병변이 초점 검출 모델에 의해 검출되었다.

이전 모든 연구에서는 아세트산 이미지만 수집되었지만, 여기서는 acetic images와 iodine images 모두 수집하여 세가지 모델을 training하여 SILs를 분류 및 세분화 하고 HSILs을 검출하여 colposcopy 검사를 보조했다.

또한 고화질 이미지가 포함된 독립적인 데이터 세트를 전체적으로 수집하여 모델의 정확도를 두 번째로 평가했다. 두 데이터셋에서 모델의 성능을 전문가와 비교했다.

본 연구의 목적은 colposcopy image에서 HSIL을 효율적이고 정확하게 인식하고 검출하는 새로운 colposcopy 진단 시스템을 구축하고 colposcopy 전문의의 진단 및 조직검사를 보조하는 것이다.

## Results

**The basic information of the modelling dataset.** 

등록 후 모델 교육 및 평가를 위해 일반 10,365건, LSIL 6,357건, HSIL 5,608건을 포함하여 22,330건이 선정되었다. 정상 사례, LSIL 사례 및 HSIL 사례의 대표 이미지는 그림 1에 나와 있다. 그림 2A에 연령 분포, HPV 감염 상태, 세포학 결과 및 TZ 유형이 나와 있다.

**The ResNet model can simply classify colposcopy images into two categories.**

분류 모델에서는 1개의 acetic image, 1개의 iodine image, 연령, HPV testing result, 세포학 결과(cytology result),  TZ type이 입력 지수로 사용되었다. Pathology diagnoses은 출력 지수로 사용되었다. 이 중 10,365건의 정상 사례, 6,357건의 LSIL 사례, 5,608건의 HSIL 사례가 8:1의 비율로 교육 세트, 테스트 세트 및 검증 세트로 비례적으로 구분되었다. 유효한 세트의 최종 결과는 표 1에 나와 있다.

분류 모형의 곡선 아래 영역(AUC)은 검증 집합에서 0.93에 도달했으며(그림 2B), 민감도는 85.38%, 특이성은 82.62%, 정확도는 84.10%였다. 또한 모형의 양의 예측값(PPV)과 음의 예측값(NPV)은 각각 85.02%와 83.03%였다.

**The U-Net model can precisely segment the lesions in the cervix.**

모두 11,198개의 acetic images과 11,198개의  iodine images가 segmentation model에 별도로 입력되었다. U-Net 모델은 annotation 후 픽셀 수준에서 교육되었으므로 분할 모델은 끝에 SIL이 될 수 있는 픽셀로 구성된 예측 영역을 출력한다. 그림 3은 접지 진실 영역(오른쪽)과 예측 영역(왼쪽)을 모두 초점 영상과 요오드 영상으로 나타냅니다. 대표적인 결과는 두 분야 간의 일관성이 높은 것으로 나타났다. 대표적인 failed image는 그림에서 볼 수 있다. S1. 누락된 병변의 대부분이 LSIL로 병적으로 진단되었으며, 병변이 잘못 진단된 이유는 명확하지 않다.

**The MASK R-CNN model can detect HSIL lesion.**

11,198 케이스의 총 22,396건의 영상이 검출 모델에 활용되었다. 그럼에도 불구하고, 초경 영상과 요오드 영상은 별도로 훈련되었습니다. 마지막으로, 여러 개의 직사각형 예측 프레임이 HSIL이 될 신뢰 계수와 함께 표시됩니다. 그림 4에 Acetic 영상과 요오드 영상의 IoU 및 예측 프레임의 평균 IoU 분포가 나와 있습니다. 조직검사 수를 제어하기 위해 최종 결과로 상위 3개의 신뢰도 HSIL 예측 프레임만 채택되었습니다. 보다 구체적으로, 고정된 직경의 원형 레이블을 사용하여 생검을 보조하는 가장 의심스러운 부위를 표시했습니다(그림 5). 대표적인 고장 영상은 그림에서 볼 수 있습니다. S2입니다.

1,120건의 사례가 있는 유효성 검사 세트의 결과는 표 2(초점 영상) 및 표 3(요오드 영상)에 나열되어 있습니다. 초심 영상과 요오드 영상에서 HSIL을 감지하기 위한 리콜은 각각 84.67%와 84.75%였습니다. HSIL의 PPV는 초산 영상에서 19.98%, 요오드 영상에서 21.22%였습니다. LSIL과 HSIL을 합치면 검출 모델은 62.09%와 64.41%의 PPV로 아세트마그와 요오드 이미지에서 각각 82.55%와 82.45%의 SIL을 리콜할 수 있습니다.

**The validation results in high-definition images.** 

선택 후 5,384개의 사례가 총 9,060개의 사례에서 독립 데이터셋에 등록되었다. 모든 이미지는 3,375개의 일반 케이스, 1,246개의 LSIL 케이스 및 763개의 HSIL 케이스를 포함한 고화질 전자 콜포스코프에 의해 촬영되었다. 연령 분포, HPV 테스트 및 세포학 결과, TZ 유형은 그림 6A에 나와 있다.

분류 모델에서 고화질 영상의 양성 사례와 음성 사례를 구분하는 민감도, 특수성, 정확도는 각각 73.37%, 58.16%, 63.83%였습니다(표 4).

PPV와 NPV는 각각 51.07%와 78.58%였으며 AUC는 0.7127이었습니다(그림 6B). 비교하기 위해, 여성 병원, 의과대학, 저장 대학 등에서 양성 및 음성 환자를 구별하기 위한 대장내시경 전문가 5명의 민감도, 특수성 및 정확성을 계산하여 표 5에 제시했습니다.

## Discussion

본 연구에서는 multimodal classification model, segmentation model, 및 etection model을 결합하여 대장내시경 영상에 대처하고 HSIL에 대한 진단 및 조직검사를 처음으로 지원하기 위한 종합적인 시스템을 구축했다. 연구에 등록된 일반 이미지는 전자 콜포스코프와 광전기 콜포스코프를 포함한 세 개의 주요 콜포스코프 브랜드에서 촬영되었습니다. 세 가지 유형의 이미지 모두 당사의 모델에 완벽한 수용성을 보였습니다. 또한, 또 다른 두 개의 전자 콜포스코프가 촬영한 고화질 이미지에서, 우리가 구축한 모델들은 주니어 전문가들과 동등한 진단 정확도에 도달할 수 있었고 HSIL을 더 잘 감지할 수 있는 능력을 보였습니다.



## Methods
### Data resource

각각의 적격 사례에 대해, 7.5배율의 식염수 이미지, 초산 이미지, 요오드 이미지를 포함한 그녀의 대장내시경 영상과 환자의 나이, HPV 검사 및 세포학 결과, TZ(변환 구역) 유형, 병리학적 진단 등을 포함한 해당 임상 데이터가 수집되었다.

연령이 다른 여성은 HPV 감염 상태와 대장균 복사 인상에 있어 서로 다른 수준의 신뢰성을 나타낼 수 있다. 그들은 또한 다른 심사 전략에도 적용됩니다32. 따라서, 환자 연령은 관리를 개선하기 위해 세 그룹으로 나눈다(표 7).

HPV 테스트 결과는 음성과 양성으로 구분되었다.

 cytology diagnoses는 6가지로 구분되었다.

TZ 유형은 국제경부병리학 및 대장내시경 연맹(IFC) 대장내시경 용어 38(표 10)에 따라 세 가지 범주로 구분되었다.

pathological diagnoses는 normal, low-grade squamous intraepithelial lesion (LSIL, including the condylomatous variant), high-grade squamous intraepithelial lesion (HSIL)으로 구분되었다.

확립된 모델을 보다 잘 평가하기 위해 대장내시경 영상의 독립적인 검증 데이터 세트와 해당 연령, HPV 검사 결과, 세포학 결과, 변환 구역의 유형 및 병리학적 진단이 수집되었습니다. 

### Data pre-process.

적격 케이스별로 1개의 Acetic 영상과 1개의 요오드 영상이 유지되고 크기가 512*512 픽셀로 조정되었다.

모델링 데이터 세트의 크기 조정된 모든 이미지는 K-means algorithm에 의해 100개의 범주로 나뉘고 8:1:1의 비율로 유효한 세트 및 테스트 세트, 세 개의 세트로 무작위로 재배치되었다.

일반 이미지, LSIL 이미지 및 HSIL 이미지는 세 세트로 균등하게 배포되도록 별도로 재배치되었다.

연령, HPV 테스트 및 세포학 결과, TZ 유형을 포함한 Text information은 표 1~4에 제시된 방법에 따라 코드화되었다. 예를 들어 HR-HPV 양성 및 ASCUS 세포학 결과를 가진 45세 환자는 3 TZ를 를 인풋하는데, 그녀의 texting code은 01001010000001이다.

### transfer learning model.

높은 효율성을 얻기 위해 1,000개가 넘는 범주의 100만 개 이상의 이미지가 포함된 ImageNet이라는 데이터베이스에서 ResNet 모델을 교육하여 사전 교육된 딥러닝 모델을 확보했다. 이를 기반으로, 미리 훈련된 ResNet 모델을 백본으로 사용하는 다중 모드 ResNet 분류 모델, U-Net42 분할 모델 및 마스크 R-CNN43 탐지 모델을 fine-tune하기 위해 대장내시경 영상이 입력되었다.

### Multi-modal ResNet classification model to simply classify the images into two groups.

두 개의 ResNet-50 모델이 각각 초산 영상과 요오드 영상에 사용되었다. Cervix 영역은 텍스트, 장비 및 비 Cervix 조직 문제와 같은 초혈 및 요오드 영상에 대한 다른 원치 않는 정보로 인해 먼저 추출되었다.

임상 진단은 종종 아세트 영상과 요오드 영상을 오랫동안 비교한 후에 이루어지므로, 훈련 과정 중에 아세트 영상 특징과 요오드 영상 특징을 융합하면 경부 병변을 더 잘 포착하고 보다 과학적인 진단을 제공할 수 있다.

마지막으로 연령, HPV 테스트 결과, 세포학 결과, TZ 타입의 코드화된 비이미지 정보를 모델에 입력하고 퓨전된 이미지 기능과 통합했습니다. 모든 영상은 자궁경부에 편평상피 내 병변(LSIL 및 HSIL 포함)이 없음을 의미하는 음의 그룹과 자궁경부에서 하나 이상의 SIL이 발견되었음을 의미하는 양의 두 그룹으로 분류된다(그림 8C).

Classification model의 경우 input image 은 더 짧은 에지에서 512로 조정되었다.  BCE loss를 10의 positive weith로 사용했다. Batch size는 16으로 설정되었다. learning rate 1e-4, weight decay 1e-4 및 momentum 0.9와 함께 SGD 최적화기가 사용되었다. 10 epochs 동안 training loss이 더 이상 줄어들지 않았을 때 학습률은 0.9로 증가했다.

### U-Net segmentation model to segment the lesions apart from the normal areas

Classification 모델과 같이,U-Net model은  transfer-leraning ResNet model을 기반으로 fine-tuned 되었다.

이미지의 크기는 512*512픽셀로 조정되었으며, 대장내시경 전문가가 작성한 주석에 따라 각 픽셀에 "병변"의 경우 "1" 또는 "정상"의 경우 "0"으로 라벨이 지정되었다. 결국 가능한 조직검사 부위를 나타내는 모든 병변이 강조 표시된다(그림 8D).

### **Mask-R-CNN detection model to offer the final HSIL biopsy sites**

Mask R-CNN 모델은 transfer-learningResNet 모델을 기반으로 기존 분할 주석의 경계 상자 설명에 따른 대장내시경 영상에서 병변 영역을 감지했다.



세 모델 모두 랜덤 색상, 랜덤 대비, 랜덤 포화, 랜덤 색 변환( random color, random contrast, random saturation, and random hue transformation)을 적용했다.

