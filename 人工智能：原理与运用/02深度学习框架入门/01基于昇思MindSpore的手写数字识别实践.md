# 基本数据结构：张量
网络运算中最基本的数据结构--张量
## 张量的概念
数据预处理的目的是把图像、文本等内容转换为模型可以计算的张量($tensor$)。在数学上是一个代数对象，而在深度学习里面，张量其实就是一个多维数组
## 张量的构建
```python
from mindspore import Tensor, dtype as mstype  

tensor_0d = Tensor(24, mstype.int32)  
tensor_1d = Tensor([2, -8, 7], mstype.int32)  
tensor_2d = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.int32)  
tensor_3d = Tensor([  
    [[5, 8, 9], [4, 7, 8]],  
    [[4, 6, 6], [3, 5, 7]],  
    [[2, 4, 6], [2, 54, 8]]  
], mstype.int32)
```
## 张量的属性
- 形状（shape）:张量的形状，是一个元组
- 数据类型（dtype）：表示张量种每一个元素的类型
- 维数（ndim）：张量的维度数
```python 
print(f"1. 3D tensor:\n {tensor_3d}")  
print(f"2. Shape of 3D tensor:\n {tensor_3d.shape}")  
print(f"3. Data type of 3D tensor:\n {tensor_3d.dtype}")  
print(f"4. Dimension of 3D tensor:\n {tensor_3d.ndim}")
```
```output
1. 3D tensor:
 [[[ 5  8  9]
  [ 4  7  8]]

 [[ 4  6  6]
  [ 3  5  7]]

 [[ 2  4  6]
  [ 2 54  8]]]
2. Shape of 3D tensor:
 (3, 2, 3)
3. Data type of 3D tensor:
 Int32
4. Dimension of 3D tensor:
 3
 ```
# 数据预处理方法
![[Pasted image 20251015144939.png]]
在数据预处理种，MindSpore提供基于Pipeline的数据引擎
## 数据集下载和加载
在 `mindspore.dataset` 提供了多种内置的数据集兼容接口
在本案例中使用 `MnistDataset`
```python 
from mindspore.dataset import MnistDataset, vision, transforms  
from download import download  
  
url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"  
  
path = download(url, "./", kind="zip",replace=True)  

train_dataset = MnistDataset('MNIST_Data/train')  
test_dataset = MnistDataset('MNIST_Data/test')
```
我们选取一张图片的内容,看一下情况
```python
image, label = next(train_dataset.create_tuple_iterator())  
print(image.shape, image.dtype, label)
```
```output
(28, 28, 1) UInt8 9
```
说明这个是一个 $28\times 28 \times 1$的张量，其中$28 \times 28$表示图片的高和宽， $1$表示图片的通道数。常见的图片类型中，灰度图的是$1$, RGB的通道是 $3$
## 数据集处理和数据增强
通常情况下，我们不直接加载原始数据进入神经网络进行训练。在这个例子中我们主要以图像操作为示例，而 `mindspore.dataset.vision` 提供了一系列的图像操作
#### 获取数据集的列名
我们有数据列，就可以在后续制定数据列进行数据预处理
```python
print(train_dataset.get_col_names())
```
```output
['image', 'label']
```
#### 图像缩放（rescale）
```python
rescale = vision.Rescale(1.0/255.0, 0)  
rescaled_image = rescale(image.asnumpy())
```
参数
- $Rescale$:缩放因子
- $Shift$：平移因子
结果
- $Ouput:image \times rescale + shift$
#### 标准化（normalize）
```python
normalize = vision.Normalize(mean=(0.1307,), std=(0.3081,))  
normalized_image = normalize(rescaled_image)
```
参数
- $Mean$:图像每个通道的均值
- $Std$:图像每个通道的标准差
- $is\_hwc$:bool值
	- $True$:$(Height,\ Width,\ Channel)$
	- $False$:$(Channel,\ Height,\ Width)$
#### HWC2CHW
```python
hwc2chw = vision.HWC2CHW()  
chw_image = hwc2chw(normalized_image)  
print(normalized_image.shape, chw_image.shape)
```
```output
(28, 28, 1) (1, 28, 28)
```
用于转换图片格式
#### 数据分批（batch）
```python
train_dataset = train_dataset.batch(batch_size=32)  
for image, label in train_dataset.create_tuple_iterator():  
    print(f"shape of image [N C H W]:{image.shape}")
```
```output
shape of image [N C H W]:(32, 28, 28, 1)
```
经过预处理后， 数据集的图片变成了四维张量分别是$(Batchsize,\ Channels,\ Height,\ Weight)$
#### 数据预处理流水线（pipeline）
![[Pasted image 20251016094556.png|350]]
```python
def datapipe(path:str, batch_size:int=32):  
    image_transforms = [  
        vision.Rescale(1.0/255.0, 0),  
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),  
        vision.HWC2CHW()  
    ]  
  
    label_transforms = transforms.TypeCast(mindspore.int32)  
  
    dataset = MnistDataset(path)  
    dataset = dataset.map(operations=image_transforms, input_columns="image")  
    dataset = dataset.map(operations=label_transforms, input_columns="label")  
    dataset = dataset.batch(batch_size=batch_size)  
  
    return dataset
    
    
train_dataset = datapipe("MNIST_Data/train", 64)  
test_dataset = datapipe("MNIST_Data/test", 64)
```
## 数据集迭代
完成数据集操作后用 `create_tuple_iterator()` 或 `create_dict_iterator()` 接口创建数据迭代器，迭代访问数据，后面进入神经网络进行训练
- `create_dict_iterator()`
```python
for data in train_dataset.create_dict_iterator():  
    print(data['image'].shape)  
    print(data['label'].shape)
```
```output
(32, 28, 28, 1)
(32,)
```
- `create_tuple_iterator()`
```python
image, label = next(train_dataset.create_tuple_iterator())  
print(image.shape)  
print(label.shape)
```
```output
(32, 28, 28, 1)
(32,)
```
# 神经网络构建的方法
## 定义模型类
mindspore 通过定义模型类来构建神经网络
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
        )  
  
  
    def construct(self, x):  
        x = self.flatten(x)  
        logits = self.dense_relu_sequential(x)  
        return logits
        
        
model = Network()  
print(model)
```
```output
Network(
  (flatten): Flatten()
  (dense_relu_sequential): SequentialCell(
    (0): Dense(input_channels=784, output_channels=512, has_bias=True)
    (1): ReLU()
    (2): Dense(input_channels=512, output_channels=512, has_bias=True)
    (3): ReLU()
    (4): Dense(input_channels=512, output_channels=10, has_bias=True)
  )
)
```
在构建网络中， 使用了 `nn.Flatten`, `nn.Dense`, `nn.ReLu`, `nn.SequentialCell` 四种网络层或框架 API
## 模型层 
我们在这里分解上面构造的网络层的每一层， 我们前面把图片数据处理成了$(64,\ 1,\ 28,\ 28)$ 的张量，作为模型的输入
#### nn.Flatten
$Flatten$ 层的作用是将数据展平， 常被放在卷积和全连接层之间，将卷积层输出的特征图转换为向量序列的形式
```python
flatten_layer = nn.Flatten()  
image_after_flatten = flatten_layer(image)  
print(image_after_flatten.shape)
```
```output
(64, 784)
```
nn.Flatten 默认从维度为1开始到最后一维进行展平。
$$
\begin{align}
(64,\ 1,\ 28,\ 28 ) \implies &(64,\ 1\times 28\times 28) \equiv (64,\ 784)
\end{align}
$$
#### nn.Dense
$Dense$ 是全连接层，使用全重和偏差进行线性变换
```python
dense_layer = nn.Dense(in_channels=784, out_channels=512, weight_init="normal", bias_init="zeros")  
image_after_dense1 = dense_layer(image_after_flatten)  
print("before dense:", image_after_flatten.shape)  
print("After dense:", image_after_dense1.shape)
```
```output
before dense: (64, 784)
After dense: (64, 512)
```
这里使用四个参数（还有其他参数没有涉及）
- $in\_channels$：输入通道
- $out\_channels$：输出通道
- $weight\_init$：权重初始化
- $bias\_init$：偏置初始化
#### nn.ReLU
[[全连接神经网络以及训练流程#ReLU 函数|ReLU]]是网络中加入的非线性的激活函数，帮助神经网络学习各种复杂的特征
```python 
relu_layer = nn.ReLU()  
image_after_relu = relu_layer(image_after_dense1[0:5])  
print(f"before ReLU:{image_after_dense1.shape[0:5]}")  
print(f"after ReLU:{image_after_relu.shape[0:5]}")
```
```output
before ReLU:(64, 512)
after ReLU:(5, 512)
```
#### nn.SequentialCell
SequentialCell 是一个有序的Cell容器，输入张量将按照定义的顺序通过所有Cell。可以通过使用nn.SequentialCell 将多个网络层快速组合构建一个神经网路模型 
```python
dense_relu_sequential = nn.SequentialCell(  
    nn.Dense(in_channels=28*28, out_channels=512),  
    nn.ReLU(),  
    nn.Dense(in_channels=512, out_channels=512),  
    nn.ReLU(),  
    nn.Dense(in_channels=512, out_channels=10)  
)  
  
image_after_sequential = dense_relu_sequential(image_after_flatten)  
print(f"Shape of image after SequentialCell:{image_after_sequential.shape}")
``` 
```output
Shape of image after SequentialCell:(64, 10)
```
#### nn.Softmax
[[全连接神经网络以及训练流程#Softmax 函数|Softmax]] 是将神经网络最后一个全连接层返回的logits的值缩放为$[0,1]$，表示每一个类别的预测概率
```python
softmax = nn.Softmax(axis=-1)  
pred_probab = softmax(logits)
```
这里使用参数$ax is$表示指定维度数值和为1
## 模型参数
模型参数（$pa rameter$）是指神经网络中虚要训练的参数矩阵和向量。在模型定义的时候采用随机初始化通常是nn层内的 weight/bias/gamma/beta 等
在MindSpore中，我们可以利用 `get_parameter` 获取网络参数
#### 模型参数获取
```python
model =  Network()  
  
for param in model.get_parameters():  
    print(param)  
    break
```
```output
Parameter (name=dense_relu_sequential.0.weight, shape=(512, 784), dtype=Float32, requires_grad=True)
```
显示的上面基于 SequentialCell 构建的第一个层级
- $name$：参数名称，第一段是参数所在的网络层名称，最后一个是参数名称
	- *示例是 SequentialCell 中第一层 Dense 的权重*
- $shape$：参数的形状
- $Dtype$：参数的数据类型
- $r equires\_{} grad$：参数是否参与梯度更新
	- $r equires\_{} grad = True$可以通过 `model.trainable_params` 获取
```python
model.trainable_params
```
```output
<bound method Cell.trainable_params of Network(
  (flatten): Flatten()
  (dense_relu_sequential): SequentialCell(
    (0): Dense(input_channels=784, output_channels=512, has_bias=True)
    (1): ReLU()
    (2): Dense(input_channels=512, output_channels=512, has_bias=True)
    (3): ReLU()
    (4): Dense(input_channels=512, output_channels=10, has_bias=True)
  )
)>
```
# 模型训练流程
对于框架而言，深度学习整体流程包含四个部分
- 通过正向计算得到 logits
- 通过[[全连接神经网络以及训练流程#损失函数|损失函数]]计算正向结果 logits 和正确标签 targets 之间的误差，也就是 loss
- 根据 loss 进行反向传播，获得整个权重对应的梯度
- 把梯度更新到网络权重上
## 单步训练逻辑及过程
一个 step 逻辑如下
```python
# 定义损失函数  
loss_fn = nn.CrossEntropyLoss(reduction="mean")  

# 定义前向传播函数  
def forward_fn(data, label):  
    logits = model(data)  
    loss = loss_fn(logits, label)  
    return loss, logits  

# 定义优化器  
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)  

# 定义梯度更新函数  
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)  

# 定义单步训练函数  
def train_step(data, label):  
    (loss, _), grads = grad_fn(data, label)  
    optimizer(grads)  
    return loss
```
#### 正向计算
```python
# 定义前向传播函数  
def forward_fn(data, label):  
    logits = model(data)  
    loss = loss_fn(logits, label)  
    return logits, loss  
```
我们直接调用模型，将数据输入模型，获得二位张量输出（logits），包含了每一个类别的原始预测值。然后利用损失函数（loss_fn）评估模型的预测值和目标值的误差。
不同的损失函数使用于不同的任务
![[Pasted image 20251110203750.png|600]]
#### 反向计算
```python
# 定义梯度更新函数  
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
```
为了优化模型参数，需要求参数对 loss 求导数，获得 function 的微分函数
#### 权重更新
```python
# 定义单步训练函数  
def train_step(data, label):  
    (loss, _), grads = grad_fn(data, label)  
    optimizer(grads)  
    return loss
```
在函数返回 loss, logits, grads 就可以将梯度值 grads 放入 optimizer 中进行权重更新，完成一个完整的步骤训练。权重更新的过程也称为模型优化（$mo d el\ optimization$）
模型优化是 MindSpore 提供的多种优化算法的实现，称为[[全连接神经网络以及训练流程#优化器|优化器]]（$optimmizer$）
优化器内部定义了模型的参数优化过程，即梯度如何更新至模型参数，在这里我们使用的是随机梯度下降（$stochastic\ gradient\ descent,\ SGD$）
```python
# 定义优化器  
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)
```
我们通过 `model.trainable_params()` 方法获取模型可以训练的参数，并传入优化器来初始化
## 数据集遍历迭代
在实现一个步骤的训练逻辑后，使用for loop 遍历数据集进行模型训练。训练过程中，一次数据集的完整遍历称为一个 epochs （即整个训练数据集通过神经网络进行一次完整的前向传播和反向传播的过程），可以通过超参 epochs 来设置模型需要经历几个 epochs 的训练
```python
def train(model, dataset):  
    size = dataset.get_dataset_size()  
    model.set_train()  
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):  
        loss = train_step(data, label)  
  
        if batch % 100 == 0:  
            loss, current = loss.asnumpy().mean().item(), batch  
  
            print(f"loss: {loss:>7f}[{current:>3d}/{size:>3d}]")  

for t in range(epochs):  
    print(f"Epoch {t+1}\n--------------------------------------------------")  
    train(model, train_dataset)  
  
print("Done!")
```
# 模型评估流程
在训练的时候会采用边训练边评估的方式，每完成一轮训练，会基于验证集对模型进行评估
差别是之前被设置为了 `model.set_train(False)`
```python
def test_loop(model, dataset, loss_fn):  
    num_batches = dataset.get_dataset_size()  
    model.set_train(False)  
    total, test_loss, correct = 0, 0, 0  
    for data, label in dataset.create_tuple_iterator():  
        pred = model(data)  
        total += len(data)  
        test_loss += loss_fn(pred, label).asnumpy().mean()  
        correct += (pred.argmax(1) == label).asnumpy().sum()  
          
    test_loss /= num_batches  
    correct /= total  
      
    print(f"Test:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
```
# 模型推理流程
## 模型的保存和加载
实际上， 在训练网络模型的过程中，我们更希望保存中间和最后的结果，用于微调和后续的模型的推理和部署
MindSpore 提供了 `save_checkpoint` 接口，通过训练好的输入模型对象即可进行保存。在后面使用模型使用 `load_checkpoint` 接口加载之前保存的 checkpoint 文件，并利用 `load_param_into_net` 将其加载到构建好的模型中，可以方便的恢复和使用
#### 模型保存
```python
mindspore.save_checkpoint(model, f"./checkpoints/model_epoch_{t+1}.ckpt")
```
#### 模型加载
```python
filename = "./checkpoints/model_epoch_3.ckpt"  
param_dict = mindspore.load_checkpoint(filename)  
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print("param_not_load:", param_not_load)  
print(f"load model successful from {filename}")
```
```output
param_not_load: []
load model successful from ./checkpoints/model_epoch_3.ckpt
```
param_not_load 为空表示所有参数加载完成
## 模型推理
过程和模型测试的内容差不多， 设置 `model.set_train(False)` 就进行输入数据
```python
model.set_train(False)  
for data, label in test_dataset:  
    pred = model(data)  
    predicted = pred.argmax(1)  
    print(f"predicted: {predicted[:10]}\nActual: {label[:10]}")  
```
```output
predicted: [8 4 1 6 3 5 7 0 3 1]
Actual: [8 4 1 6 3 5 4 0 3 1]
```