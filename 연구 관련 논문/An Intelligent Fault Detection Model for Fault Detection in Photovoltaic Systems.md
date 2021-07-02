# An Intelligent Fault Detection Model for Fault Detection in Photovoltaic Systems

[태양광발전시스템 고장감지를 위한 지능형 고장탐지모델]

2020년 9월 발표

  PV system에서 효과적인 고장 진단은 다른 환경 상태에서 current/voltage(I/V) parms를 이해하는 것이 중요하다. 특히 겨울철에는 PV 시스템에서 특정 결함 상태의 I/V characters가 정상 상태의 I/V characters와 매우 유사하다. 따라서 정상적인 고장 감지 모델은 정상 작동하는 PV 시스템을 결함 상태로 잘못 예측할 수 있으며, 그 반대의 경우도 마찬가지다. 이 논문에서는 PV 시스템의 고장 감지 및 분류를 위한 지능형 고장 진단 모델을 제안한다. 실험 검증(experimental verification)을 위해 다양한 고장 상태 및 정상 상태 데이터 세트가 광범위한 환경 조건에서 겨울철에 수집된다. 수집된 데이터셋은 몇 가지 데이터 마이닝 기법을 사용하여 정규화하고 사전 처리한 다음 probabilistic neural network(PNN)에 공급된다. PNN 모델은 새 데이터를 가져올 때 결함을 예측하고 분류하기 위해 과거 데이터로 교육된다. 기계 학습의 다른 분류 방법과 비교할 때, 훈련 받은 모델은 예측 정확도에서 더 나은 성능을 보였다.

## Introduction

  Fault detection 및 적시 문제 해결(timely troubleshooting)은 태양광 발전(PV) 시스템을 포함한 모든 발전 시스템의 최적 성능을 위해 필수적입니다. 특히, 모든 상업용 발전소의 목표는 전력 생산 극대화, 에너지 손실 및 유지관리 비용 최소화, 시설 안전 운영이다. PV 시스템은 다양한 고장과 실패가 발생하기 때문에 그러한 고장과 실패의 조기 감지는 목표를 달성하는 데 매우 중요하다[1-3]. 미국 국가 전기 법규는 특정 고장으로부터 보호하기 위해 PV 설비에 OCPD(과전류 보호 장치) 및 GFDI(접지 고장 감지 인터럽터)를 설치해야 한다. 그러나 2009년 베이커스필드 화재 사례와 2011년 Mount Holly는 이러한 장치가 특정 시나리오에서 고장을 감지할 수 없음을 보여준다[4]. PV 시스템의 고장은 물리적, 환경적 또는 전기적 조건에서 발생할 수 있다. PV array fault detection를 위한 광범위한 기술이 존재하며, 가능한 솔루션을 제공하기 위해 광범위한 연구가 수행되었다. PV 시스템의 성능을 결정하는 데 가장 중요한 두 가지 파라미터는 전류(current)와 전압(voltage)이다. I/V curves에 유도된 변형을 고려하여 결함이 있는 각 모듈과 어레이의 전기적 시그니처(electrical signature)를 고정하는 간단한 current-voltage analysis method가 제안되었다. 또 다른 연구는 mismatch fault의 정량적 정보를 추출하기 위해 PV 시스템의 전기(electrical) 및 열(thermal) 모델이 결합된 적외선 온도 측정기의 사용을 보여준다. 유사한 연구는 PV blocks의 손상 감지를 위한 항공 적외선 온도 측정(aerial infrared thermography) 및 PV 시스템 효율성 평가를 위한 현장 적외선 온도 측정 기법(onfield infrared thermography-sensing technique)을 적용한 것을 보여준다. 마찬가지로 PV 시스템의 고장 감지에 반사측정법(reflectometry methods)도 사용되었다. 단락(short circuit: 전선이 붙어버린 현상) 및 절연결함(insulation defects)를 위한 TDR(Time Domain Reflectometry) 방법이 사용되었으며, 최근에는 PV시스템에서 지락(ground faults: 접지 사고)과 노후된 임피던스(교류회로에서 전류가 흐르기 어려운 정도) 변화를 검출하기 위한 spread spectrum TDR (SSTDR) 방법이 조사되었다. 그 외에도, 아크 결함(arc faults)을 감지하기 위한 wavelet decomposition techniques 및  line-line 결함 검출을 위한 multiresolution signal decomposition의 적용은 문헌에서도 확인할 수 있다. 최근 논문에서는 PV 시스템의 몇 가지 진보된 fault detection approaches에 대한 포괄적인 연구를 제공했다. 이 연구는 fault detection approaches을 model-based difference measurement(MBDM), real-time difference measurement(RDM),output signal analysis(OSM) 및 machine learning techniques(MLT)으로 나누었다. 또한 이러한 진보된 기술과 기존 방법을 비교하여 장단점을 제시했다. 

 오늘날 대부분의 PV 시스템은 모니터링 시스템과 함께 구축되며 대용량 기간별 데이터가 지속적으로 백업된다. 인공지능(AI) 방법은 데이터 기반이며 PV 시스템에서 빅데이터를 사용할 수 있게 되면서 이 분야에 대한 연구가 탄력을 받고 있는 것으로 보인다. 특히, Machine Learning(ML-) 기반 알고리즘과 기술이 제안되며, 여기서 모델은 결함을 예측하고 분류하기 위해 과거 데이터로 교육된다. 최근 연구에 따르면 PV 모듈의 결함 분류( fault classification)를 위한 열전도(thermography)와 ML 기법이 적용된 것으로 보고되었다. 그 연구에서는 다양한 결점 열 패널 (fault panel thermal images)의 특징을 연구하기 위해 텍스처 특성(texture feature) 분석을 채택했으며 개발된 알고리즘은 93.4% 정확도로 교육되었다. 또 다른 연구는 PV 시스템에서 결함 감지, 분류 및 localization을 위한 ML 기법의 적용을 보고한다. 그 연구에서는 100%의 예측 정확도로 알고리즘을 개발했다고 주장한다. 마찬가지로, 또다른 연구는 wavelet-based approach 및 radial basis function networks (RBFN)을 활용하여 인버터의 단락(circuit) 및 단선 고장(open circuit faults)을 감지한다. 그 연구들은 1kW single-phase stand-alone PV시스템에서 테스트 했을 때, 100% training 효율성과 97% test 효율성을 보여준다.

 ML기술을 사용하는 PV 시스템을 위한 훈련된 모델의 성능은 새 데이터를 다른 환경 조건, 특히 겨울철의 데이터로부터 가져오는 경우 크게 달라질 수 있다. 겨울철 조사(irradiation) 수준은 여름보다 훨씬 낮으며, 연구는 이러한 낮은 조사(irradiation) 수준에서 발생하는 결함이 감지되지 않은 상태로 남아있을 가능성이 더 높다고 보여주었다. 이러한 감지되지 않은 faults는 상당한 양의 전원 손실(power losses)과 패널 품질 저하를 야기하거나 심지어 패널 열화(deterioration)를 초래할 수 있다.  우리는 고장 모듈을 감지하고 모든 환경 조건에 적용되는 고장 유형을 추가로 분류하기 위한 지능형 고장 진단 모델을 제안한다. 모델은 MLP를 사용하고 Supervised Learning approach를 따른다. 다양한 환경 조건, 특히 겨울에 초점을 맞춘 다양한 결함 및 정상 상태의 과거 데이터로 강력하게 훈련된다. 데이터는 전라북도에 위치한 1.8kW의 송전망 연결(grid-connected) PV 시스템에서 수집되었다. 

 본 문서의 나머지 부분은 다음과 같이 구성되어 있다. 섹션 2는 PV 시스템 고장의 개요를 소개합니다. 섹션 3은 고장 진단 모델(fault diagnosis model)의 전체 시스템 아키텍처를 설명한다. 섹션 4는 실험 결과를 제시하고, 모델을 기존 분류 방법과 비교하며, 기타 관련 문제를 논의한다. 마지막으로 섹션 5는 article을 요약하고 마무리한다.

## Overview of PV System Faults

