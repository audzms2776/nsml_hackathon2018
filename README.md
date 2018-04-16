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

히든 레이어에서 네트워크 연산 시간이 오래 걸려서 DCBiLSTM layer를 1층만 연결하고 1D average-pooling을 하여 사용하였습니다. 
