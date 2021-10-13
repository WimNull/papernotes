# GAN论文笔记

## GAN分类
![gan_taxonomy](images/gan_taxonomy.jpg)

## [Generative Adversarial Networks(GAN-2014)](https://arxiv.org/pdf/1406.2661.pdf)
### 算法原理  
![gan_value_function](images/gan_vf.png)
&emsp;基于minimax对抗原理, GAN网络由生成器和判别器两部分组成。输入数据 $x \sim p_{data}$ , 对于生成器$G$, 给定分布$z \sim p_z$, 经过判别器后得到一个结构类似$x$的输出且$G(z) \sim p_g$, 而$p_g$则去逼近输入数分布$p_{data}$。对于判别器$D$, $D(x/G(z))$表示为真实(fake/valid)图片的概率。  
&emsp;训练生成器: $\ _{G}^{min} = log(1-D(G(z)))$  
&emsp;训练判别器: $\ _{D}^{max} = log(x)+log(1-D(G(z)))$   
&emsp;最终达到纳什均衡即完成训练(判别器正确率为50%，判断对一半fake，一半valid)，总体算法如下:
![algorithm](images/gan_algorithm.png)

## [Conditional Generative Adversarial Nets(CGAN-2014)](https://arxiv.org/pdf/1411.1784.pdf)

### 算法原理  
![cgan_value_function](images/cgan_vf.png)
<div align=center>
<img src="images/cgan_net.png" />
</div>



