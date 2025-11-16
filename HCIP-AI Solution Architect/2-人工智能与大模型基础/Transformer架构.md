#Transformer #Word2Vec #GloVe
# Attention 机制
## 为什么要引入Attention 机制
- Attention：注意力。 
- 当我们用CNN模型识别图像时，如何让模型知道图像中不同局部信息的重要性呢？ 
- 当我们用RNN模型去处理NLP相关任务时，长距离“记忆”问题如何解决呢

Attention是挑重点，就算文本比较长，也能从中间抓住重点，不丢失重要的信息。图中底部有颜色的文字就是被挑出来的重点。
## Attention机制 
- 注意力机制：在解码器的每个步骤中，使用与编码器的直接连接来聚焦于源序列的特定部分。 
- 注意力机制在输入信息中聚焦于更为关键的信息，降低对其他信息的关注度，甚至过滤掉无关信息，就可以解决信息过载问题，并提高任务处理的效率和准确性。
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=83&rect=291,451,488,586&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.83|400]]
让Encoder编码出的c向量跟Decoder解码过程中的每一个输出进行加权运算，在Decoder的每一个过程中调整权重取到不一样的
# Transformer 详解
## Transformer 的诞生
2017年，谷歌机器翻译团队发表的《Attention is All You Need》中，提出了Transformer，完全抛弃了 RNN （Recurrent Neural Network ， 循环神经网络 ） 和 CNN （Convolutional Neural Networks，卷积神经网络）等网络结构，而仅仅采用Attention机制来进行机器翻译任务，并且取得了很好的效果，注意力机制也成为了大家的研究热点

很多研究已经证明了Transformer提取特征的能力是要远强于LSTM（Long Short Term Memory，长短期记忆网络）的
## Transformer 模型结构
[transformer 结构](https://zhuanlan.zhihu.com/p/338817680)
- Transformer由Encoder和Decoder两部分组成。
- Encoder包含N个相同的layer，layer指的就是图中左侧的单元。最左边有个 “𝑁 ×”，指的是N个，如果N=6的话， 对应的就是6个。
	- Encoder中的layer结构都是相同的，但他们没有共享参数。每个layer都可以分解成两个sub-layer。每个sub-layer的输出维度d_model=512。
- Decoder也包含N=6个相同的layer
- 从编码器输入的句子首先会经过一个自注意力（Self-Attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。
- 自注意力层的输出会传递到前馈（feed-forward）神经网络中。每个位置的单词对应的前馈神经网络都完全一样（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。
- 解码器中也有编码器的自注意力（Self-Attention）层和前馈（feed-forward）层。除此之外，这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和 seq2seq模型的注意力作用相似）。
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=86&rect=79,451,275,641&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.86|450]]
## Transformer工作流程
#### 获取输入
Transformer中的输入X是由单词Embedding和位置编码（Positional Encoding） 相加得到的。最终，每个单词都被嵌入为一个512维的向量。
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=87&rect=95,458,472,584&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.87|600]]
- 每一行是一个单词的表示X。
###### 单词Embedding
单词的Embedding有多种方式可以得到，例如可以采用Word2Vec、GloVe等算法预训练得到，也可以在Transformer中训练得到。
![[语言模型定义及发展#静态词向量]]
###### 位置编码
对于Transformer来说，由于句子中的词语都是同时进入网络进行处理，顺序信息在输入网络时就已丢失。因此，Transformer需要额外的处理来告知每个词语的相对位置。其中的一个解决方案， 就是论文中提到的位置编码（Positional Encoding），将能表示位置信息的编码添加到输入中， 让网络知道每个词的位置和顺序。
$$
\begin{gather}
PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{\frac{2i}{dmoel}}} \right) \\
PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{\frac{2i}{dmoel}}} \right)
\end{gather}
$$
使用这种公式计算PE有以下的好处： 
- 使PE能够适应比训练集里面所有句子更长的句子。 
- 可以让模型容易地计算出相对位置，对于固定长度的间距$k$，$PE{(pos+k)}$可以用 $PE(pos)$计算得到。
#### 将输入传入Encoder
将得到的单词表示向量矩阵传入Encoder 中，经过6个Encoder block后可以得到句子所有单词的编码信息矩阵C。
每一个Encoder block输出的矩阵维度与输入完全一致
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=90&rect=285,450,462,633&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.90|400]]
- $X_{n\times d}$ 输入矩阵：$n$是句子中单词数量，$d$表示向量的维度，论文中d=512。
#### 将编码矩阵传递给Decoder
将Encoder输出的编码信息矩阵C传递到Decoder中， Decoder依次会根据当前翻译过的1~i的单词翻译第i+1 个单词。
在整个过程中 ， 翻译到第 i+1个单词的时候，需要通过Mask（掩盖）操作遮盖住第i+1之后的单词。
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=91&rect=193,451,479,642&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.91|600]]
## 自注意力机制
#### Attention 计算过程
###### 第一步：生成查询向量、键向量和值向量
![[HCIP-AI-Ascend Developer V2.0 培训教材 .pdf#page=238&rect=91,455,484,623&color=red|HCIP-AI-Ascend Developer V2.0 培训教材 , p.238]]
###### 第二步：计算得分
*Query和每一个key计算的得出相似性得分s*
![[HCIP-AI-Ascend Developer V2.0 培训教材 .pdf#page=239&rect=84,472,482,628&color=red|HCIP-AI-Ascend Developer V2.0 培训教材 , p.239]]
###### 第三步：归一化
![[HCIP-AI-Ascend Developer V2.0 培训教材 .pdf#page=240&rect=87,450,482,634&color=red|HCIP-AI-Ascend Developer V2.0 培训教材 , p.240]]
*8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值。*
###### 第四步：对加权值向量求和
![[HCIP-AI-Ascend Developer V2.0 培训教材 .pdf#page=241&rect=80,446,488,635&color=red|HCIP-AI-Ascend Developer V2.0 培训教材 , p.241]]
#### Self-Attention
自注意力机制的目标是：**让序列中每个词根据整个序列的信息，生成新的上下文相关表示**
通俗理解：
- 每个词都可以“关注”序列中的其他词
- 输出的词向量不仅包含自身信息，还包含与其他词的关系
```ad-example
下列句子是我们想要翻译的输入句子： 
> The animal didn’t cross the street because it was too tired. 

这个“it”在这个句子是指什么呢？它指的是“street”还是“animal”呢？这对于人类来说是一个简单的问题，但是对于算法则不是。
当模型处理这个单词“it”的时候，自注意力机制会允许“it”与“animal”建立联系。
```
## 自注意力机制的强大作用
随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码。
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=93&rect=105,459,275,621&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.93|400]]
## 多头注意力机制
论文为了进一步完善自注意力层，增加一种多头注意力机制（Multi-head Attention）。
多头注意力是多组自注意力构成的组合，自注意力机制能帮助建立包括上下文信息的词特征表达，多头注意力能帮助学习到多种不同类型的上下文影响情况。
```ad-example
“今天阳光不错，适合出去跑步”，在不同情景下，“今天”同“阳光”、“跑步”的相关性是不同的。
```
头越多，越有利于捕获更大更多范围的相关性特征，增加模型的表达能力
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=94&rect=161,474,391,620&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.94|600]]在两个方面提高了注意力层的性能：
- 扩展了模型专注于不同位置的能力；
- 给出了注意力层的多个“表示子空间”。

