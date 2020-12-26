**텐서플로우와 머신러닝으로 시작하는 자연어처리 첫번째 책은 아래 링크를 참고해주시기 바랍니다.** 

**첫번째 책 링크: https://github.com/NLP-kr/tensorflow-ml-nlp**

***공지: GPT2 모델 다운로드 링크가 변경될 예정입니다. 모델 다운로드는 다음과 같이 받아주시기 바랍니다.**
```
wget https://github.com/NLP-kr/tensorflow-ml-nlp-tf2/releases/download/v1.0/gpt_ckpt.zip -O gpt_ckpt.zip
```


# NLPBOOK

텐서플로2와 머신러닝으로 시작하는 자연어처리(로지스틱회귀부터 BERT와 GPT2까지)  
<p align="center">
  <img src="main.png" width="450" height="500" /> 
</p>

## 소개 (Introduction)

책에 수록된 자연어 처리 예제들을 모아놓은 리파지토리입니다.

본 리파지토리는 텐서플로우와 머신러닝으로 시작하는 자연어처리 책과 같이 활용하여 공부하시면 더욱 도움이 되실겁니다.

## 설치방법 (Environments)
```
conda create -n {사용할 환경 이름} python=3.6
conda activate {사용할 환경 이름}
pip install -r requirements.txt
```
<!--** 추가로 본 실습에서는 `tensorflow==2.2.0` 환경에서 작동이 가능한 것을 테스트 했습니다.-->

만약 설치가 정상적으로 진행되지 않으신다면 python 3.6을 설치하여 진행해주시기 바랍니다.
```
conda install python=3.6
```
<!-- #### GPU 사용 시 CUDA 설치 관련 -->

<!-- - GPU를 사용하는 경우에는 텐서플로우와 호환이 되는 CUDA Version을 맞춰 설치해야 합니다. -->
<!-- - 현재 본 프로젝트는 `tensorflow==1.10` 버전에서 실행이 가능하도록 구현 및 테스트를 하였습니다. -->
<!-- - `tensorflow-gpu==1.10` 의 경우 `CUDA 9.0`을 설치해주시기 바랍니다. -->

<!-- >> - `tensorflow-gpu>=1.13` 의 경우 `CUDA 10.0`을 설치해주시기 바랍니다. -->
<!-- >> - `tensorflow-gpu>=1.5,<=1.12` 의 경우 `CUDA 9.0`을 설치해주시기 바랍니다. -->
<!-- >> - `tensorflow-gpu>=1.0,<=1.4` 의 경우 `CUDA 8.0`을 설치해주시기 바랍니다. -->

## Jupyter Docker 실행

Docker 환경 사용시 19.03 이후 버전을 사용하길 권장합니다.

- `bash build_jupyter_cpu.sh` 또는 `bash build_jupyter_gpu.sh`를 실행하면 docker image을 생성합니다.
- `bash exec_jupyter_cpu.sh` 또는 `bash exec_jupyter_gpu.sh`를 실행하면 docker환경에서 jupyter가 실행됩니다.
-  jupyter 실행 포트번호는 8889 이므로 해당 포트번호에 대해서 사용이 가능해야 합니다.

## 목차 (Table of Contents)

**준비 단계** - 자연어 처리에 대한 배경과 개발에 대한 준비를 위한 챕터입니다.

1. [들어가며](./1.Intro)
2. [자연어 처리 개발 준비](./2.NLP_PREP)
3. [자연어 처리 개요](./3.NLP_INTRO)

**자연어 처리 기본** - 자연어 처리에 기본적인 모델에 대한 연습 챕터입니다.

4. [텍스트 분류](./4.TEXT_CLASSIFICATION)
5. [텍스트 유사도](./5.TEXT_SIM)

**자연어 처리 심화** - 챗봇 모델을 통해 보다 심화된 자연어 처리에 대한 연습 챕터입니다.

6. [챗봇 만들기](./6.CHATBOT)
7. [미세 조정 학습](./7.PRETRAIN_METHOD)

## Colab 실습

Colab 실습은 7장에 한하여 별도 저장소를 공개하였습니다. 

- [저장소 링크](https://github.com/NLP-kr/tensorflow-ml-nlp-tf2-colab)

## 문의사항 (Inquiries)
[Pull Request](https://github.com/NLP-kr/tensorflow-ml-nlp-tf2/pulls)는 언제든 환영입니다.
문제나 버그, 혹은 궁금한 사항이 있으면 [이슈](https://github.com/NLP-kr/tensorflow-ml-nlp-tf2/issues)에 글을 남겨주세요.

**이슈를 확인하기 전에 Wiki에 도큐먼트 먼저 보시고 이슈에 글을 남겨주세요!

## 저자 (Authors)
ChangWookJun / @changwookjun (changwookjun@gmail.com)  
Taekyoon  / @taekyoon (tgchoi03@gmail.com)  
JungHyun Cho  / @JungHyunCho (reniew2@gmail.com)  
Ryan S. Shin / @aiscientist (sungjin7127@gmail.com)
