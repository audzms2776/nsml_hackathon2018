# nsml_hackathon2018

## https://github.com/naver/ai-hackathon-2018
## https://hack.nsml.navercorp.com/

## nsml 해커톤 로컬용 모델 개발

## my_moview_review
네이버 영화 평점 예측 미션 랭킹
pytorch 사용
LSTMCell과 cnn을 사용하였습니다. 

## mymy_kin
네이버 지식iN 질문 유사도 미션 랭킹 <br/>
tensorflow 사용 <br />
Densely Connected Bidirectional LSTM with Applications to Sentence Classification (https://arxiv.org/pdf/1802.00889.pdf) <br/>
네트워크를 구현하였습니다. <br/>
### 이미지
![dcbilstm image](https://github.com/audzms2776/nsml_hackathon2018/blob/master/dcbilstm.PNG "https://arxiv.org/pdf/1802.00889.pdf")

konlpy를 이용하여 문장에서 명사, 동사 중에서 같은 단어를 제거하였습니다. <br/>
문장의 길이를 더 짧게하여 학습하였습니다. <br/>
같은 네트워크 모델을 사용하였을 때보다 성능 향상이 있었습니다. <br/>
