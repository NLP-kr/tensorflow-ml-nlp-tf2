# 8. GPT3

GPT3에서 활용해볼 수 있는 퓨샷 러닝과 피-튜닝을 실습 해봅시다.
이 실습에서는 7장에서 활용한 GPT2를 가지고 퓨샷 러닝과 피-튜닝 실습합니다.

실습 자원의 없는 경우 [Colab 실습 저장소의 실습 자료](https://github.com/NLP-kr/tensorflow-ml-nlp-tf2-colab)를 이용해주시거나, 
해당 폴더에 있는 실습 파일을 Colab에서 실행해주시기 바랍니다.

## 실습 자원

본 장에서 활용하게 될 모델과 데이터 셋 입니다.

### 모델

- [Huggingface의 Transformers 라이브러리 도큐먼트](https://huggingface.co/transformers/)
- [Ko-GPT2 모델 깃허브 저장소](https://github.com/SKT-AI/KoGPT2)

- **중요**: GPT2 모델 저장소가 변경 되었습니다. 모델 다운로드는 다음 방식으로 다운로드 받으시면 됩니다. 
   - 기존 GPT2 모델 링크는(Dropbox 저장소) 더 이상 지원하지 않습니다.
```
wget https://github.com/NLP-kr/tensorflow-ml-nlp-tf2/releases/download/v1.0/gpt_ckpt.zip -O gpt_ckpt.zip
```

## 실습 내용

### 8.3 퓨샷 러닝
- [퓨샷 러닝을 활용한 텍스트 분류](./8.3.gpt2_fewshot_NSMC.ipynb)

### 8.4  피-튜닝
- [피-튜닝을 활용한 텍스트 분류](./8.4.gpt2_p_tuning_NSMC.ipynb)
