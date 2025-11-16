import mindspore
from mindspore import nn
import mindspore.dataset.vision as vision
from mindspore.dataset import MnistDataset, transforms

# 定义模型
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


    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

# 数据加载
def datapipe(path: str, batch_size: int = 32):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]

    label_transforms = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(operations=image_transforms, input_columns="image")
    dataset = dataset.map(operations=label_transforms, input_columns="label")
    dataset = dataset.batch(batch_size=batch_size)

    return dataset

model = Network()
train_dataset = datapipe("MNIST_Data/train", 64)
test_dataset = datapipe("MNIST_Data/test", 64)
# 超参设置
learning_rate = 0.0001
epochs = 3

# 定义损失函数
loss_fn = nn.CrossEntropyLoss(reduction="mean")

# 定义前向传播函数
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# 定义优化器
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

# 定义梯度更新函数
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 定义单步训练函数
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = float(loss.asnumpy().mean()), batch

            print(f"loss: {loss:>7f}[{current:>3d}/{size:>3d}]")


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



for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------------------------")
    train(model, train_dataset)
    test_loop(model, test_dataset, loss_fn)
    mindspore.save_checkpoint(model, f"./checkpoints/model_epoch_{t+1}.ckpt")

print("Done!")

# filename = "./checkpoints/model_epoch_3.ckpt"
# param_dict = mindspore.load_checkpoint(filename)
# param_not_load, _ = mindspore.load_param_into_net(model, param_dict)

# print("param_not_load:", param_not_load)
# print(f"load model successful from {filename}")

# model.set_train(False)
# for data, label in test_dataset:
#     pred = model(data)
#     predicted = pred.argmax(1)
#     print(f"predicted: {predicted[:10]}\nActual: {label[:10]}")
#     break
