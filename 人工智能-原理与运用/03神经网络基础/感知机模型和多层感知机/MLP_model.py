import mindspore as ms
from mindspore import Tensor, nn
from mindspore.train import Model as Model
from mindspore.train.callback import LossMonitor
import mindspore.dataset as ds
from mindspore.nn import Accuracy
import numpy as np


class MLPModel(nn.Cell):
    def __init__(self):
        super(MLPModel, self).__init__()
        # 输入到隐藏层
        self.fc1 = nn.Dense(in_channels=2, out_channels=4)
        self.relu = nn.ReLU()
        # 隐藏层到输出层
        self.fc2 = nn.Dense(in_channels=4, out_channels=1)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 定义简单数据集
X_train = np.array([[1.2, 0.7],
                    [-0.3, -0.5],
                    [3.0, 0.1],
                    [-0.1, -1.0],
                    [0.5, 0.6],
                    [-0.7, -0.8],
                    [2.5, 0.4],
                    [-0.2, -1.3]], dtype=np.float32)

Y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32).reshape(-1, 1)

# 转换为 Tensor
X_train = Tensor(X_train)
Y_train = Tensor(Y_train)

# 创建训练数据集
train_dataset = ds.NumpySlicesDataset({"features": X_train, "labels": Y_train}, shuffle=True)
train_dataset = train_dataset.batch(4)

# 实例化模型
net = MLPModel()

# 定义损失函数和优化器
criterion = nn.BCELoss(reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)

# 创建模型对象
model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={'accuracy':Accuracy()})

# 训练模型
model.train(10, train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)