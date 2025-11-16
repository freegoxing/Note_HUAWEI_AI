# 计算图类型
## 动态图
其核心是计算图的构建和计算同时发生（Define by Run）
#### 原理 
类似 python 解释器，在计算图中定义一个张量，其值就已经被计算且确定了
#### 优点
Pythonic 语法，在调试模型时比较方便，能够实时得到中间结果的值
#### 缺点
由于所有解点都需要保存，导致难以对整个计算图进行优化
## 静态图
其核心是计算图的构建和实际计算分开（define and run）
*以 Tensor Flow 的思路为主*
#### 原理 
在构建阶段，根据完整的计算流程对原始的计算图进行优化和调整，编译到更省内存和计算量更少的计算图。编译之后的图结构不在改变，所以称为“静态图”。在计算阶段，根据输入数据执行编译好的计算图，从而得到结果
#### 优点
比起动态图，对全局的信息掌握更丰富，可作的优化也会跟多
#### 缺点
中间过程对用户来说是个黑盒，用户无法像使用动态图一样实时拿到结果
# 运行时切换
Context 模式是一种全局的设置模式。在构建网络之前进行配置，默认情况下采用动态图模式（pynative_mode），通过调用 `mindspore.set_context(mode=mindspore.GRAPH_MODE)` 切换为全局静态图模式
```python
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
```
# 即时编译（Just In Time, JIT）
结合动静态图的优缺点，为了保留动态图的灵活性的同时利用静态图的优势，MindSpore采用了即时编译的范式。具体来说将 train_step 构建为一个 function
- 如果希望采用动态图，就直接执行该 function 即可；
- 如果希望采用静态图或进行编译优化，只需要添加 `mindspore.jit` 修饰器，通过一行代码切换动静态图
```python
@ms.jit  
def train_step(data, label):  
    (loss, _), grads = grad_fn(data, label)  
    optimizer(grads)  
    return loss
```
# 局部静态加速
采用 `Cell.construct` 添加 `mindspore.jit` ：单我们为神经网络的某个部分进行加速时，可以在 Construct 方法上面使用 `mindspore.jit` 修饰器。在调用实例化的时候，该模块会自动被编译为静态图
```python
class Network(nn.Cell):  
    def __init__(self):  
        super().__init__()  
        self.flatten = nn.Flatten()  
        self.dense_relu_sequential = nn.SequentialCell(  
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),  
            nn.ReLU(),  
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),  
            nn.ReLU(),  
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros"),  
            # nn.Softmax(axis=-1)  
        )  
  
    @ms.jit  
    def construct(self, x):  
        x = self.flatten(x)  
        logits = self.dense_relu_sequential(x)  
        return logits
```

