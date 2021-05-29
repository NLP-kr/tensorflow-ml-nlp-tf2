# 5. 텍스트 유사도

앞서 텍스트 분류 모델을 다뤄 보았다면 이제 서로 다른 텍스트가 얼마나 의미적으로 가까운지도 한 번 알아 봅시다.

여기서는 텍스트 유사도 모델을 만들어 각 문장 간의 거리가 어떻게 되는지 실습해보고자 합니다.

이 장은 영어 데이터셋만을 가지고 실습을 하게 됩니다.

## 데이터

데이터는 캐글(Kaggle)의 "Quora Question Pairs"를 가지고 실습하게 됩니다.

## 실습 내용

- [5.2 데이터 분석과 전처리](./5.2.EDA&preprocessing.ipynb)

### 5.3 모델링
- [XG 부스트 텍스트 유사도 분석 모델](./5.3.1.XGboost.ipynb)

- [CNN 텍스트 유사도 분석 모델](./5.3.2.Quora_CNN.ipynb)

- [MaLSTM - 모델 구현](./5.3.3_Quora_LSTM.ipynb)


<!--
## QuoraQuestionPairs (Link 정리 해서 공유)

* [QuoraQuestionPairs](https://github.com/changwookjun/Kaggle/tree/master/QuoraQuestionPairs)   
  + [Understanding LSTM](https://github.com/changwookjun/Kaggle/blob/master/QuoraQuestionPairs/Understanding%20LSTM%20Networks.ipynb)  
  + [History Word Vectors](https://github.com/changwookjun/Kaggle/blob/master/QuoraQuestionPairs/The%20Amazing%20Power%20Of%20Word%20Vectors.ipynb)  
  + [QuoraQuestionPairsAnalysis.ipynb](https://github.com/changwookjun/Kaggle/blob/master/QuoraQuestionPairs/QuoraQuestionPairsAnalysis.ipynb)  
  + [Kaggle Quora Question Pairs MaLSTM Paper.ipynb](https://github.com/changwookjun/Kaggle/blob/master/QuoraQuestionPairs/Kaggle%20Quora%20Question%20Pairs%20MaLSTM%20Paper.ipynb)    
  + [Kaggle Quora Question Pairs MaLSTM Source.ipynb](https://github.com/changwookjun/Kaggle/blob/master/QuoraQuestionPairs/Kaggle%20Quora%20Question%20Pairs%20MaLSTM%20Source.ipynb)  
-->
