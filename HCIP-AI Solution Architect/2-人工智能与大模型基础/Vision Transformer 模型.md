#ViT
## ViT 诞生背景
- 在2020年至2021年期间，卷积神经网络主导了CV领域，而Transformer则成为了NLP领域的标准配置。许多研究致力于将NLP领域的思路推广到CV领域，主要可以分为两类： 
	- 第一，将注意力机制与CNN相结合
	- 第二，在整体结构不变的情况下，用注意力机制替换CNN中的某些结构。例如，将自注意力机制（Self-Attention）应用于语义分割领域，产生了诸如Non-local、DANet、CCNet等优秀的语义分割模型。
- 然而，这些方法仍然需要依赖于CNN来提取图像特征。因此，研究人员开始思考是否可以完全不依赖CNN，直接使用Transformer作为编码器来提取图像特征。 
- ViT就是在思考如何将Transformer结构扩展到CV领域。
## Vision Transformer
ViT（Vision Transformer）算法中尝试将标准的Transformer结构直接应用于图像，并对整个图像分类流程进行最少的修改。具体来讲，ViT算法中，会将整幅图像拆分成小图像块，然后把这些小图像块的线性嵌入序列作为Transformer的输入送入网络，然后使用监督学习的方式进行图像分类的训练。
作为CV领域最经典的Transformer算法之一，不同于传统的CNN算法，ViT尝试将标准的Transformer结构直接应用于图像，并对整个图像分类流程进行最少的修改。
为了满足Transformer输入结构的要求，
- 将整幅图像拆分成小图像块(Patch Embedding)
- 然后把这些小图像块的线性嵌入序列输入到网络
- 使用了Class Token的方式进行分类预测。
```ad-note
title:Class Token
假设我们将原始图像切分成共9个小图像块，最终的输入序列长度却是10，也就是说我们这里人为的增加了一个向量进行输入，我们通常将人为增加的这个向量称为 Class Token。那么这个 Class Token 有什么作用呢？

我们可以想象，如果没有这个向量，也就是将9个向量（1~9）输入 Transformer 结构中进行编码，我们最终会得到9个编码向量，可对于图像分类任务而言，我们应该选择哪个输出向量进行后续分类呢？

因此，ViT算法提出了一个可学习的嵌入向量 Class Token( 向量0)，将它与9个向量一起输入到 Transformer 结构中，输出10个编码向量，然后用这个 Class Token 进行分类预测即可。
```
## Vision Transformer 变体
ViT参考BERT，共设置了三种模型变体
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=118&rect=111,493,445,551&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.118]]
- Layers就是 Transformer Encoder中重复堆叠Encoder Block的次数
- Hidden Size就是对应通过 Embedding层后每个token的dim（向量的长度
- MLP size是Transformer Encoder 中MLP Block第一个全连接的节点个数（是Hidden Size的四倍）
- Heads代表 Transformer中Multi-Head Attention的heads数。
## Vision Transformer 实验效果
该算法在中等规模（例如ImageNet）以及大规模（例如ImageNet-21K、JFT-300M）数据集上进行了实验验证，发现：
- Transformer相较于CNN结构，缺少一定的平移不变性和局部感知性，因此在数据量不充分时， 很难达到同等的效果。具体表现为使用中等规模的ImageNet训练的Transformer会比ResNet 在精度上低几个百分点。
- 当有大量的训练样本时，结果则会发生改变。使用大规模数据集进行预训练后，再使用迁移学习的方式应用到其他数据集上，可以达到或超越当前的SOTA水平。
```ad-note
title:SOTA(State of the Art)
SOTA：指在某一领域中，使用最新技术或方法达到的最佳性能表现。它通常是指最先进的技术或方法，用于解决特定问题或任务。在深度学习中，SOTA是指使用最新算法、模型架构和数据增强技术等手段，在特定任务领域中取得最好的性能表现。
```

