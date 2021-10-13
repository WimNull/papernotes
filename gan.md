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

&emsp;除了网络的输入有区别，其他地方与GAN一样

## [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks(LAPGAN)](https://arxiv.org/pdf/1506.05751.pdf)

### 图像金字塔
高斯金字塔(降采样): (1)对图像进行高斯核卷积, (2)将偶数行除去
拉普拉斯金字塔: (1)降采样然后上采样, (2)原图与重建图像差异: $L_i = G_i - Up(Down(Gi))$

### 算法原理 
1 用途：生成高品质的自然图片 
2 创新点：利用拉普拉斯金字塔，由粗到精逐级生成越发清楚的图像。 
3 突破：GAN只能生成低像素的图片，而LAPGAN可以生成高像素的图片。 
4 本质：用CGAN生成拉普拉斯金字塔。 
训练过程: 
![lapgan_gennet](images/lapgan_train.png)
使用推理过程: 
![lapgan_gennet](images/lapgan_infer.png)
