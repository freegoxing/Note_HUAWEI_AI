# 函数式自动微分
![500](assets/04昇思MindSpore的高阶操作/Pasted%20image%2020251112210332.png)
这个是对输入的$x$,进行操作输出logits 
$$
\begin{gather}
log its = x \cdot w +b \\
loss = y-log its
\end{gather}
$$
```python
import mindspore as ms  
from mindspore import ops, Tensor  
  
# 定义计算函数  
def function(x, y, w, b):  
    z = ops.matmul(x, w) + b  
    loss = ops.binary_cross_entropy_with_logits(z, y)  
    return loss  
  
# 准备输入张量  
x = Tensor([[0.5, 0.3]], ms.float32)    # shape (1, 2)  
y = Tensor([[1.0]], ms.float32)         # 标签  
w = Tensor([[0.2], [0.8]], ms.float32)  # shape (2, 1)  
b = Tensor([0.1], ms.float32)           # shape (1,)  
  
# 创建梯度函数，指定对 w 和 b 求梯度  
grad_fn = ms.value_and_grad(function, grad_position=(2, 3))  
  
# 调用求值与求梯度  
loss, (grad_w, grad_b) = grad_fn(x, y, w, b)  
  
print("Loss:", loss)  
print("Grad w:", grad_w)  
print("Grad b:", grad_b)
```
在这里我们指定了参数位置
- $grad\ \ position$：表示对对应位置进行求导
- *示例中是（2,3）表示对 function 输入的第二，第三个参数求导即 $\partial\ loss/\partial\ w$ 和 $\partial\ loss/\partial\ b$*