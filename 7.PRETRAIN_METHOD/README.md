# 7. 사전 학습 모델

사전 학습 (Pretrained) 모델과 미세 조정 (Fine-tuned) 학습에 대해서 알아보도록 합시다.

여기서 활용하는 모델은 BERT라는 모델과 GPT2라는 모델을 활용합니다.

실습 자원의 없는 경우 [Colab 실습 저장소의 실습 자료](https://github.com/NLP-kr/tensorflow-ml-nlp-tf2-colab)를 이용해주시기 바랍니다.

## 실습 자원

본 장에서 활용하게 될 모델과 데이터 셋 입니다.

### 모델

- [Huggingface의 Transformers 라이브러리 도큐먼트](https://huggingface.co/transformers/)
- [구글 공식 BERT 깃허브 저장소](https://github.com/google-research/bert)
- [Ko-GPT2 모델 깃허브 저장소](https://github.com/SKT-AI/KoGPT2)

- **중요**: GPT2 모델 저장소가 변경 되었습니다. 모델 다운로드는 다음 방식으로 다운로드 받으시면 됩니다. 
   - 기존 GPT2 모델 링크는(Dropbox 저장소) 더 이상 지원하지 않습닏..
```
wget https://github.com/NLP-kr/tensorflow-ml-nlp-tf2/releases/download/v1.0/gpt_ckpt.zip -O gpt_ckpt.zip
```


### 데이터

- [KorNLU Dataset](https://github.com/kakaobrain/KorNLUDatasets)
- [Naver NLP Challenge Dataset](https://github.com/monologg/korean-ner-pytorch/tree/master/data)
- [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/)

## 실습 내용

### 7.2 버트를 활용한 미세 조정 학습
- [버트를 활용한 한국어 텍스트 분류 모델](./7.2.1.bert_finetune_NSMC.ipynb)
- [버트를 활용한 한국어 자연어 추론 모델](./7.2.2.bert_finetune_KorNLI.ipynb) ([데이터 분석](./7.2.2.KorNLI_EDA.ipynb))
- [버트를 활용한 한국어 개체명 인식 모델](./7.2.3.bert_finetune_NER.ipynb) ([데이터 분석](./7.2.3.NER_EDA.ipynb))
- [버트를 활용한 한국어 텍스트 유사도 모델](./7.2.4.KorSTS_EDA.ipynb) ([데이터 분석](./7.2.4.bert_finetune_KorSTS.ipynb))
- [버트를 활용한 한국어 기계 독해 모델](./7.2.5.bert_finetune_KorQuAD.ipynb) ([데이터 분석](./7.2.5.KorQuAD_EDA.ipynb))

### 7.3 GPT

### 7.4 GPT2를 활용한 미세 조정 학습

- [GPT2를 활용한 한국어 언어 생성 모델](./7.4.1.gpt2_finetune_LM.ipynb)
- [GPT2를 활용한 한국어 텍스트 분류 모델](./7.4.2.gpt2_finetune_NSMC.ipynb)
- [GPT2를 활용한 한국어 자연어 추론 모델](./7.4.3.gpt2_finetune_KorNLI.ipynb)
- [GPT2를 활용한 한국어 텍스트 유사도 모델](./7.4.4.gpt2_finetune_KorSTS.ipynb)
