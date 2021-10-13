# GAN论文笔记

## GAN分类
![gan_taxonomy](images/gan_taxonomy.jpg)

## [Generative Adversarial Networks(Original GAN)](https://arxiv.org/abs/1406.2661)
### 算法原理  
&emsp;基于minimax对抗原理, GAN网络由生成器和判别器两部分组成。输入数据 $x \sim p_{data}$ , 对于生成器$G$, 给定分布$z \sim p_z$, 经过判别器后得到一个结构类似$x$的输出且$G(z) \sim p_g$, 而$p_g$则去逼近输入数分布$p_{data}$。对于判别器D, $D(x/G(z))$表示为真实(fake/valid)图片的概率。  
&emsp;训练生成器: $V^{min}_{\ G}=log(1-D(G(z)))$  
&emsp;训练判别器: $V^{max}_{\ D}=log(x)+log(1-D(G(z)))$   
&emsp;最终达到纳什均衡即完成训练(判别器正确率为50%，判断出一半为fake，一半为valid)，总体算法如下:
![algorithm](images/gan_algorithm.png)
