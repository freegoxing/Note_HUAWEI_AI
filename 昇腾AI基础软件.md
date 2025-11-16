# 昇腾AI基础软件
## CANN
CANN （ Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，对上支持多种 AI框架，对下服务AI处理器与编程， 发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台
*对标的是NVIDA的CUDA（Compute Unified Device Architecture）、AMD的ROCm（Radeon Open Compute platform）*
#### CANN的总体架构
一共有5层结构
![](assets/昇腾AI基础软件/Pasted%20image%2020251101104348.png)
## MindSpore
MindSpore作为新一代深度学习框架，是源于全产业的最佳实践，最佳匹配昇腾处理器算力，支持终端、边缘、云全场景灵活部署，开创全新的AI编程范式，降低AI开发门槛
*旨在实现易开发、高效执行、全场景统一部署三大目标*
- MindSpore作为全场景AI框架，所支持的有端（手机与IOT设备）、边（基站与路由设备）、云（服务器）场景的不同系列硬件
- MindSpore上训练出来的模型文件，可通过Serving部署在云服务中执行，也可通过Lite执行在服务器、端侧等设备上。同时Lite支持通过独立工具convert进行模型的离线优化，实现推理时框架的轻量化以及模型执行高性能的目标
- MindSpore抽象各个硬件下的统一算子接口，因此，在不同硬件环境下，网络模型的编程代码可以保持一致。同时加载相同的模型文件，在MindSpore支持的各个不同硬件上均能有效执行推理。
- 推理方面考虑到大量用户使用C++/C编程方式，提供了C++的推理编程接口
- 通过提供第三方硬件的自定义离线优化注册，第三方硬件的自定义算子注册机制，实现快速对接新的硬件，同时对外的模型编程接口以及模型文件保持不变。
#### 层次结构
MindSpore向用户提供了3个不同层次的API，支撑用户进行AI应用（算法/模型）开发，从高到低分别为
###### High-Level Python API
提供了训练推理的管理、混合精度训练、调试调优等高级接口，方便用户控制整网的执行流程和实现神经网络的训练推理及调优
###### Medium-Level Python API
提供网络层、优化器、损失函数等模块， 用户可通过中阶API灵活构建神经网络和控制执行流程，快速实现模型算法逻辑。
###### Low-Level Python API
主要包括张量定义、基础算子、自动微分等模块，用户可使用低阶API轻松实现张量定义和求导计算
## Mind X
Mind X包含多个组件
#### Mind DL
Mind DL（昇腾深度学习组件）是支持Atlas训练卡、推理卡的深度学习组件。提供了昇腾AI处理器的调度
#### Mind Edge
Mind Edge（昇腾智能边缘组件）提供边缘AI业务容器的全面生命周期管理能力，同时提供严格的安全可性保障
#### Mind SDK
Mind SDK：使用华为提供的SDK和应用案例快速开发并部署人工智能应用
#### ModelZoo
提供了丰富的各领域SOTA模型，一键下载，同时提供全流程开发工具，支持使用ModelShow直观比较
# 昇腾AI大模型开发套件
## MindFormers
MindFormer套件基于MindSpore内置的并行技术和组件化设计，目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类训练模型和SOTA下游任务应用
## ModelLink
ModelLink旨在为华为昇腾芯片上的大语言模型提供端到端的解决方案，包括模型、算法、以及下游任务
- 制作预训练数据集/制作指令微调数据集
- 预训练/全参微调/低参微调
- 推理（人机对话）
- 评估基线数据集（Benchmark）
- 使用加速特性（加速算法+融合算子）
# 昇腾AI大模型开发加速库
## MindSpeed
MindSpeed是专为昇腾设备设计的大模型训练加速库。它不仅考虑到了硬件层面的亲和性，还结合了昇腾平台特有的软硬件特性在并行计算、模型切分、集合通信等方面作了优化相比于DeepSpeed而言，MindSpeed兼容原生Megatron-LM框架，适配特性如下
- 张量并行
- 流水线并行
- 序列并行
- 重计算
- 分布式优化器
- 异步分布式数据并行
#### Ascend Extension for PyTorch
Ascend Extension for PyTorch 插件是基于昇腾的深度学习适配框架 ， 使昇腾 NPU 可以支持 PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。
![[HCIP-AI-Ascend Developer V2.0 培训教材 .pdf#page=349&rect=242,473,478,633&color=red|HCIP-AI-Ascend Developer V2.0 培训教材 , p.349|500]]
###### MindSpeed LLM
LLM模型套件，提供LLM大模型端到端训练流程，提供模型结构增强、微调、偏好对齐、数据工程能力，覆盖业内主流语言大模型。 
###### MindSpeed MM
多模态模型套件，聚焦多模态生成、多模态理解，提供多模态大模型端到端训练流程，提供多模数据预处理、编码、融合、训练微调等能力，覆盖业内主流多模态大模型。 
###### MindSpeed Core
昇腾加速库，提供计算、显存、通信、并行四个维度的优化，支持长序列、MoE等关键特性的性能加速。 
###### Ascend Extension for PyTorch（即torch_npu插件）
昇腾PyTorch适配插件，继承开源PyTorch特性，针对昇腾AI处理器系列进行深度优化，支持用户基于PyTorch框架实现模型训练和调优。 
###### PyTorch原生库/三方库适配
适配支持PyTorch原生库及主流三方库，补齐生态能力， 提高昇腾平台易用性。
# 昇腾AI大模型全流程工具链
## MindStudio
![](assets/昇腾AI基础软件/Pasted%20image%2020251101111452.png)
#### MindStudio Insight
是一款调优可视化工具，集成了CANN数据的分析和可视化等功能。
# 昇腾推理引擎
## MindIE
MindIE（Mind Inference Engine，昇腾推理引擎）是华为昇腾针对AI全场景业务的推理加速套件
![[HCIP-AI Solution Architect V1.0 培训教材.pdf#page=513&rect=183,461,365,633&color=yellow|HCIP-AI Solution Architect V1.0 培训教材, p.513|500]]
![600](assets/昇腾AI基础软件/Pasted%20image%2020251101111935.png)
#### MindIE-RT：昇腾AI处理器推理加速引擎
#### MindIE-ATB：基于Transformer结构的加速库
#### MindSpore+MindIE：图编译/分布式并行/模型压缩
#### PyTorch+MindIE：支持训练模型平滑迁移推理
MindIE Torch是针对PyTorch框架模型，开发的推理加速插件。PyTorch框架上训练的模型利用MindIE Torch提供的简易C++/Python接口，少量代码即可完成模型迁移， 实现高性能推理。MindIE Torch向下调用了MindIE RT组件能力。
#### MindIE-LLM：大语言模型推理模型套件
#### MindIE-SD：视图生成加速框架分层
MindIE SD是MindIE的视图生成推理模型套件，其目标是为稳定扩散（Stable Diffusion, SD）系列大模型推理任务提供在昇腾硬件及其软件栈上的端到端解决方案
#### MindIE-Service：昇腾服务化框架
提供推理服务化部署和运维能力
- MindIE-Server：推理服务端，提供模型服务部署能力，支持命令行部署RESTful服务
- MindIE-Client：提供服务客户端标准API，简化服务调用
- MindIE-MS：服务策略管理，提供服务运维能力

