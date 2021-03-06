# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse

import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

from my_movie_review.dataset import MovieReviewDataset


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)


class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """

    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.max_length, 200, self.embedding_dim, batch_first=True)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, data: list):
        """

        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if torch.cuda.is_available():
            data_in_torch = data_in_torch.cuda()

        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        out = embeds.view((batch_size, self.embedding_dim, 200))
        h0 = Variable(torch.zeros(self.embedding_dim, batch_size, 200).type(torch.cuda.FloatTensor))
        c0 = Variable(torch.zeros(self.embedding_dim, batch_size, 200).type(torch.cuda.FloatTensor))
        out, _ = self.lstm(out, (h0, c0))

        re_out = out[:, -1, :]
        hidden = torch.sigmoid(self.fc1(re_out))
        hidden = torch.sigmoid(self.fc2(hidden))
        output = hidden * 9 + 1

        return output


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=8)
    config = args.parse_args()

    DATASET_PATH = './'

    model = Regression(config.embedding, config.strmaxlen)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)

        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)
                label_vars = Variable(torch.from_numpy(labels))

                if torch.cuda.is_available():
                    label_vars = label_vars.cuda()

                loss = criterion(predictions, label_vars)

                if torch.cuda.is_available():
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Batch : ', i + 1, '/', total_batch,
                      ', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]
            print('epoch:', epoch, ' train_loss:', float(avg_loss / total_batch))
