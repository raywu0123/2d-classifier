from math import ceil

from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelBase
from .fc import FC_NET


class OCSVM(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oc_svm = OneClassSVM()
        self.model = FC_NET(
            layer_num=3,
            units=20,
            **kwargs,
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def fit(self, x, y, label):
        data = np.asarray([x, y]).transpose()
        data_train, data_test, label_train, label_test = train_test_split(data, label)

        print(min(label), max(label))
        print(data_train.shape, data_test.shape, label.shape)

        self.oc_svm.fit(data_train)
        print('finish training oc_svm.')

        for i_epoch in range(self.num_epochs):
            log = self._fit_epoch(data_train, label_train)
            if i_epoch % 10 == 0:
                print(f'epoch: {i_epoch}')
                print(log)
                log = self._validate(data_test, label_test)
                print(log)
            self._save_entropy_distribution()

    def predict(self, x, y):
        data = np.asarray([x, y]).transpose()

        svm_scores = self.oc_svm.score_samples(data).reshape(-1, 1)
        self.model.eval()
        num_batch = ceil(len(data) / self.batch_size)
        pred_buff = []
        for i in range(num_batch):
            end_index = min(len(data), (i + 1) * self.batch_size)
            batch_data = data[i * self.batch_size: end_index]
            batch_data_tensor = torch.Tensor(batch_data)
            logits = self.model(batch_data_tensor)

            svm_score = torch.Tensor(svm_scores[i * self.batch_size: end_index])
            logits = logits + torch.max(svm_score) - svm_score
            pred = F.softmax(logits, dim=-1)
            pred = pred.data.numpy()
            pred_buff.extend(pred)

        self.model.train()
        return np.asarray(pred_buff)

    def _calc_entropy(self, x, y):
        original_shape = x.shape
        x = x.reshape(-1)
        y = y.reshape(-1)
        p = self.predict(x, y)
        entropy = np.sum(- p * np.log(p + 1e-12), axis=-1)
        entropy = entropy.reshape(original_shape)
        return entropy
