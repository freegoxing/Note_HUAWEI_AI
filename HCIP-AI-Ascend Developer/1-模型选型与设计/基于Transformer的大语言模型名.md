#Transformer #GPT #BERT  #Llama #DeepSeek
# [[Transformer架构#Transformer 详解|Transformer 详解]] 
## 序列模型
对应之前图像任务，输入数据基本上没有关联，即前面的数据对后面数据的判断没有影响。但是对于很多视频、文本等数据，前后是有联系的，对后面数据的判读需要前面的数据支持。而CNN 在解决这类问题上效果不佳
RNN等一系列算法被提出，这类模型是在预测后面数据时，会把前面的数据加权与当前数据作为此刻的输入，保证前面信息的传递，增加后续数据预测的准确性
但是这类算法有一些缺陷：并行程度不高，导致训练时间长；序列过长，会导致前面数据遗忘， 因此数据记忆能力较弱
## [[Transformer架构#Transformer 模型结构|Transformer模型结构]]
## [[Transformer架构#Transformer工作流程|Transformer工作流程]]
## [[Transformer架构#自注意力机制|Transformer自注意力机制]]
# 基于Transformer的大语言模型
## GPT
## BERT
## Llama
## DeepSeek
详见[[DeepSeek解析]]
