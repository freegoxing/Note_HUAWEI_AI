#Transformer  #GPT #BERT #Few-Shot #Zero-Shot #One-Shot #RLHF 
BERT与GPT-1采用相同的两阶段任务：首先预训练阶段，先是通过语言模型任务预训练模型；然后再根据具体的任务进行Fine-tuning模型。
```ad-note
title:Fine-tuning模型
**大模型微调**是通过特定领域的数据对预训练模型进行进一步训练，以优化其在特定任务上的表现。其核心目的是赋予模型定制化功能，并使其学习领域知识，从而更好地适应特定任务。
```
# BERT概述及架构
## BERT 的结构图
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=97&rect=96,454,466,631&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.97|700]]
## 输入部分的处理
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=98&rect=92,510,462,625&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.98]]
## Fine-tuning 阶段
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=100&rect=104,473,454,598&color=red|HCIP-AI Solution Architect V1.0 培训教材, p.100]]
# GPT概述及架构
## 产生背景
NLP（Natural Language Processing，自然语言处理）领域中只有小部分标注过的数据， 而有大量的数据未标注，所以为了充分利用大量未标注的原始文本数据，需要利用无监督学习来从文本中提取特征，最经典的例子莫过于词嵌入技术。 
词嵌入只能处理word-level级别的任务（同义词等），没法解决句子、句对级别的任务 （翻译、推理等）。
为了解决以上问题，OpenAI提出了GPT（Generative Pre-trained Transformer，通用预训练Transformer）框架，用一种通用的大模型来完成语言理解任务。
GPT的训练过程分为两个阶段：
- Pre-training
- Fine-tuning
目的是在于学习一种通用的Representation方法，针对不同种类的任务只需略作修改便能适应。
## GPT-1
- 使用Transformer而不使用LSTM在于，虽然预训练有助于捕获一些语言信息，但 LSTM模型限制了他们在短期内的预测能力。相反，Transformer的选择使我们能够捕获更长远的语言结构，同时在其他不同的任务中泛化性能更好。
- GPT-1使用的是Transformer的Decoder框架。
- GPT-1保留了Decoder的Masked Multi-Attention层和Feed Forward层，并扩大了网络的规模。将层数扩展到12层，GPT-1还将Attention的维数扩大到768（原来为 512），将Attention的头数增加到12层（原来为8层），将Feed Forward层的隐层维数增加到3072（原来为2048）
## GPT-2
相较于GPT-1的区别
- 主推Zero-shot
	- GPT-2的核心思想就是，当模型的容量非常大且数据量足够丰富时，仅仅靠语言模型的学习便可以完成其他有监督学习的任务，不需要在下游任务微调。
```ad-note
![[提示工程#零样本提示（Zero-Shot Prompting）]]
```
## GPT-3
较于之前的GPT版本，变化点如下：
- 下游任务：相比于Zero-shot，GPT-3采用Few-shot。
```ad-note
 在few-shot learning中，提供若干个示例和任务描述供模型学习。one-shot learning 是提供1个示例和任务描述。zero-shot则是不提供示例，只是在测试时提供任务相关的具体描述。作者对这3种学习方式分别进行了实验，实验结果表明，三个学习方式的效果都会随着模型容量的上升而上升，且**few shot > one shot > zero shot**。
```
- 型结构：GPT-3延续使用GPT模型结构，但是引入了Sparse Transformer中的sparse attention模块（稀疏注意力）。
```ad-note
title:稀疏注意力(sparse attention)
稀疏注意力的核心思想就是：
- 不是每个 token 都要看所有其他 token，  
- 让每个 token 只关注 **部分重要的 token**。
```
- sparse attention与传统self-attention（称为dense attention）的区别在于：
	- dense attention：每个token之间两两计算attention，复杂度$0(n^2)$。
	- sparse attention：每个token只与其他token的一个子集计算attention，复杂度 $O(n\log n)$。 
	- 但是批判性的角度来讲，肯定是有缺点的，NLP语言中内容都是有上下文关系的，如此依赖必定会对长文本建模的效果变差。
## GPT-3.5
- GPT-3纵然很强大，但是对于人类的指令理解得不是很好，这也就延伸出了GPT-3.5即 InstructGPT诞生的思路。
- InstructGPT采用基于人类反馈的强化学习来不断微调预训练语言模型，旨在让模型能够更好地理解人类的命令和指令含义，如生成小作文、回答知识问题和进行头脑风暴等。 
- 关于InstructGPT的技术方案，原文分为了三个步骤： 
	- 基于GPT-3有监督微调
	- 奖励模型训练 
	- 强化学习训练
## ChatGPT
整体技术路线上 ， ChatGPT 在效果强大的 GPT-3.5 大规模语言模型 （ LLM ， Large Language Model）基础上，引入“人工标注数据+强化学习”（RLHF，Reinforcement Learning from Human Feedback，这里的人工反馈其实就是人工标注数据）来不断 Fine-tune预训练语言模型。
相对于InstructGPT，其不同之处在于数据是如何设置用于训练（以及收集）的， ChatGPT添加了一些对话任务的有监督数据进行微调。
## GPT-4
GPT-4是一个多模态模型：它既能接收图像输入也能接收文本输入，这使它能够描述特殊图像中的含义、从屏幕截图中总结文本以及回答包含图表的问题。
