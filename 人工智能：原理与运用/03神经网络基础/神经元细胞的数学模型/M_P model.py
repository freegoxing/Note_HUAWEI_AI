import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, nn
import numpy as np


class MPModel(nn.Cell):
    def __init__(self):
        super(MPModel, self).__init__()
        # 定义权重和阈值
        self.weights = Tensor(np.array([[1.0], [1.0]]), ms.float32)
        self.threshold = 2.0
        self.matmul = ops.MatMul()

    def construct(self, x):
        # 计算加权和
        weighted_summ = self.matmul(x, self.weights)
        # 应用阈值
        output = (weighted_summ >= self.threshold).astype(np.float32)
        return output


# 定义输入和输出
inputs = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), ms.float32)
labels = Tensor(np.array([0, 0, 0, 1]), ms.float32)
# 实例化模型
model = MPModel()
# 测试
outputs = model(inputs)
for i, input in enumerate(inputs.asnumpy()):
    print(f"Input:{input}, output:{outputs[i].asnumpy()}")
