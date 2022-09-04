# Self-Supervised Learning 
## 目标
自监督训练-->在下游任务微调(Fine tune)

## 自监督学习涉及内容

1. 辅助任务(Pretext Task)：辅助任务是可以认为是一种为达到特定训练任务而设计的间接任务，并不是真正的任务，只是用来学习更好的特征表示。pretext任务的好处是为了简化原任务的求解。主要的辅助任务包括：图像着色、图像修复等、图像补丁。
2. 下游任务(Downstream Task)：下游任务的作用之一是在得到自监督训练好的模型后，用带标签的数据进行有监督学习，来评价模型的正确率，因为数据的ground-truth仍是由人工标签来确定的，但是这些有监督训练会带有一定的约束，如少量的训练样本或较短的训练时间等。作用之二是转移自监督模型学到的特征，将其用于其他任务，只需要少量的学习（迁移学习）就可以在自监督模型基础上开展新的应用。
3. 损失函数(Loss function)。通常使用的是infoNCE。
## 发展

### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
自监督任务：1)Mask token prediction; 2)Next sentence prediction
<font color='red'> 其他不是CV重点，这不在详述 </font>

### 