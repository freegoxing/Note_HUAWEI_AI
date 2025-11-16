import numpy as np
import mindspore as ms
from mindspore import Tensor, nn

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.faltten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )


    def construct(self, x):
        x = self.faltten(x)
        logits = self.dense_relu_sequential(x)

        return logits


model = Net()
input = Tensor(np.ones([64, 1, 28, 28])).astype(np.float32)
output = model(input)
print(output)

