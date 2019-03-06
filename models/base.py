from abc import ABC
from math import ceil

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from sklearn.model_selection import train_test_split


class ModelBase(ABC):
    def __init__(self, num_class: int = 2, **kwargs):
        self.num_class = num_class

        self.model = None
        self.opt = None
        self.scheduler = None

        self.num_epochs = 200
        self.batch_size = 64
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.view_range = 5.
        x = np.linspace(-self.view_range, self.view_range, 40)
        y = np.linspace(-self.view_range, self.view_range, 40)
        self.X, self.Y = np.meshgrid(x, y)
        self.ims = []
        self.Zs = []
        self.fig, self.ax = plt.subplots()

    def fit(self, x, y, label):
        data = np.asarray([x, y]).transpose()
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-12)
        data_train, data_test, label_train, label_test = train_test_split(data, label)

        for i_epoch in range(self.num_epochs):
            log = self._fit_epoch(data_train, label_train)
            if i_epoch % 5 == 0:
                # import ipdb; ipdb.set_trace()
                print(f'epoch: {i_epoch}')
                # print(self.model.out_layer.mean)
                print(log)
                log = self._validate(data_test, label_test)
                print(log)
            self._save_entropy_distribution()
        print('complete fitting model')

    def _validate(self, data, label):
        pred = self.predict(data[:, 0], data[:, 1])
        pred = np.argmax(pred, axis=-1)
        acc = np.mean(pred == label)
        if self.scheduler is not None:
            self.scheduler.step(acc)
        return {'accuracy': acc}

    def predict(self, x, y):
        self.model.eval()
        data = np.asarray([x, y]).transpose()
        num_batch = ceil(len(data) / self.batch_size)
        pred_buff = []
        for i in range(num_batch):
            end_index = min(len(data), (i + 1) * self.batch_size)
            batch_data = data[i * self.batch_size: end_index]
            batch_data = torch.Tensor(batch_data)
            pred = self.model(batch_data)
            pred = F.softmax(pred, dim=-1)
            pred = pred.data.numpy()
            pred_buff.extend(pred)

        self.model.train()
        return np.asarray(pred_buff)

    def _fit_epoch(self, x, y):
        losses = []
        for i_batch in range(len(x) // self.batch_size):
            batch_x = x[i_batch * self.batch_size: (i_batch + 1) * self.batch_size]
            batch_y = y[i_batch * self.batch_size: (i_batch + 1) * self.batch_size]

            batch_x = torch.Tensor(batch_x)
            batch_x.requires_grad_()
            batch_y = torch.Tensor(batch_y).long()

            batch_pred = self.model(batch_x)

            loss = self.loss_fn(batch_pred, batch_y)
            loss.backward()
            self.opt.step()
            losses.append(loss.item())

        return {'loss': np.mean(losses)}

    def show_entropy_distribution_animation(self, data: tuple):
        print('generating animation')
        normed_x = (data[0] - np.mean(data[0], axis=0)) / (np.std(data[0], axis=0) + 1e-12)
        normed_y = (data[1] - np.mean(data[1], axis=0)) / (np.std(data[1], axis=0) + 1e-12)
        self.data = (normed_x, normed_y, data[2])
        animation.FuncAnimation(
            self.fig,
            self._animate,
            len(self.Zs),
            interval=1,
            blit=False
        )
        plt.show()

    def _animate(self, idx):
        self.ax.clear()
        self.ax.scatter(self.data[0], self.data[1], c=self.data[2])
        self.ax.contourf(self.X, self.Y, self.Zs[idx], alpha=.5, cmap=plt.cm.hot)
        self.ax.set_title(f'epoch: {idx}')

    def _save_entropy_distribution(self):
        Z = self._calc_entropy(self.X, self.Y)
        Z = np.exp(Z)
        self.Zs.append(Z)

    def _calc_entropy(self, x, y):
        original_shape = x.shape
        x = x.reshape(-1)
        y = y.reshape(-1)
        p = self.predict(x, y)
        entropy = np.sum(- p * np.log(p + 1e-12), axis=-1)
        entropy = entropy.reshape(original_shape)
        return entropy

    @staticmethod
    def _compute_jacobian(inputs, outputs):
        return torch.stack(
            [grad(
                [outputs[:, i].sum()],
                [inputs],
                retain_graph=True,
                create_graph=True,
            )[0]
             for i in range(outputs.size(1))
            ],
            dim=-1
        )
