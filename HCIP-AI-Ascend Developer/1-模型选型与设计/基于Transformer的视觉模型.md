#ViT 
## ViT结构
ViT的第一步是将整幅图像划分为许多大小相同的小块。 
每一个小块会被“展平”，也就是说，把原本二维的像素信息转变为一维的向量，并通过线性变换转换成固定长度的表示向量。在这些图像块向量被送入Transformer之前，ViT还会额外添加一个特殊的标记，称为“Class Token”。 （*见[[Vision Transformer 模型#Vision Transformer|Vision Transformer]]*）
ViT利用自注意力机制（Self-Attention）来理解图像各区域之间的关联。 
经过多层Transformer的编码之后，所有图像块的信息都会在Class Token中逐渐汇聚和融合。最终，ViT会依据这个Class Token的输出，完成整张图像的分类任务。
![[HCIP-AI-Ascend Developer V2.0 培训教材 .pdf#page=279&rect=231,452,443,562&color=red|HCIP-AI-Ascend Developer V2.0 培训教材 , p.279]]