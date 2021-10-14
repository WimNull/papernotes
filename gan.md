# GAN论文笔记

## GAN分类
![gan_taxonomy](images/gan_taxonomy.jpg)

## [Generative Adversarial Networks(GAN-2014)](https://arxiv.org/pdf/1406.2661.pdf)
### 算法原理  
![gan_value_function](images/gan_vf.png)
&emsp;基于minimax对抗原理, GAN网络由生成器和判别器两部分组成。输入数据 $x \sim p_{data}$ , 对于生成器$G$, 给定分布$z \sim p_z$, 经过判别器后得到一个结构类似$x$的输出且$G(z) \sim p_g$, 而$p_g$则去逼近输入数分布$p_{data}$。对于判别器$D$, $D(x/G(z))$表示为真实(fake/valid)图片的概率。  
&emsp;训练生成器: $\ _{G}^{min} = log(1-D(G(z)))$  
&emsp;训练判别器: $\ _{D}^{max} = log(x)+log(1-D(G(z)))$   

&emsp;最终达到纳什均衡即完成训练(判别器正确率为50%, 判断对一半fake, 一半valid), 总体算法如下:
![algorithm](images/gan_algorithm.png)

## [Conditional Generative Adversarial Nets(CGAN-2014)](https://arxiv.org/pdf/1411.1784.pdf)

### 算法原理  
![cgan_value_function](images/cgan_vf.png)
<div align=center>
<img src="images/cgan_net.png" />
</div>

&emsp;除了网络的输入有区别, 其他地方与GAN一样

## [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks(LAPGAN-2015)](https://arxiv.org/pdf/1506.05751.pdf)

### 图像金字塔
高斯金字塔(降采样): (1)对图像进行高斯核卷积, (2)将偶数行除去
拉普拉斯金字塔: (1)降采样然后上采样, (2)原图与重建图像差异: $L_i = G_i - Up(Down(Gi))$

### 算法原理 
1 用途：生成高品质的自然图片  
2 创新点：利用拉普拉斯金字塔, 由粗到精逐级生成越发清楚的图像。  
3 突破：GAN只能生成低像素的图片, 而LAPGAN可以生成高像素的图片。  
4 本质：用CGAN生成拉普拉斯金字塔。  
训练过程: 
![lapgan_gennet](images/lapgan_train.png)
使用推理过程: 
![lapgan_gennet](images/lapgan_infer.png)


## [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks(DCGAN-2016)](https://arxiv.org/pdf/1511.06434.pdf)

没有很大的改进, 应用于Large-scale Scene Understanding (LSUN)
• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).  
• Use batchnorm in both the generator and the discriminator.  
• Remove fully connected hidden layers for deeper architectures.  
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.  
• Use LeakyReLU activation in the discriminator for all layers.  


## [Energy-based Generative Adversarial Network(EBGAN-2017)](https://arxiv.org/pdf/1609.03126.pdf)

### 算法原理 

![ebgan_net](images/ebgan_net.png)

**repelling regularizer(排斥正则化)**  
S为自编码器中编码得到的向量
$$
f_{P T}(S)=\frac{1}{N(N-1)} \sum_{i} \sum_{j \neq i}\left(\frac{S_{i}^{\top} S_{j}}{\left\|S_{i}\right\|\left\|S_{j}\right\|}\right)^{2}
$$

**Objective functional:**  
$$
L_D(x,z) = D(x) + [m-D(G(z))]^+
\newline
L_G(z) = D(G(z))+f_{P T}(S)
$$
其中 $[·] = max(0, \ ·)$,  EBGAN改动了discriminator(auto-encoder), 鉴别器不再鉴别输入来自于$p_{data}$还是$p_g$, 而是去鉴别图像的重构性高不高。原始discriminator的目的是学会寻找$p_data$与$p_g$之间的差异进而给图像质量打分，现在不再通过寻找差异来打分，而是使用一种“一种强烈的记忆”仅仅记住$p_data$的形状，然后对于一个任意的输入x，只要x符合这个记忆的样子就给高分，反之给低分。
&emsp;训练方法: 先可以预训练这个自编码器，然后生成器使用自编码器与排斥正则化进行优化训练

## [BEGAN: Boundary Equilibrium Generative Adversarial Networks(BEGAN-2017)](https://arxiv.org/pdf/1703.10717.pdf)
主要贡献为提出Boundary Equilibrium GAN, 结构为一个生成器和自编码器如EBGAN图: 
$$
\begin{cases}\mathcal{L}_{D}=\mathcal{L}(x)-k_{t} \cdot \mathcal{L}\left(G\left(z_{D}\right)\right) & \text { for } \theta_{D} \\ \mathcal{L}_{G}=\mathcal{L}\left(G\left(z_{G}\right)\right) & \text { for } \theta_{G} \\ k_{t+1}=k_{t}+\lambda_{k}\left(\gamma \mathcal{L}(x)-\mathcal{L}\left(G\left(z_{G}\right)\right)\right) & \text { for each training step } t\end{cases}
\newline
\mathcal{M}_{global} = \mathcal{L}(x) + |\gamma \mathcal{L}(x) - \mathcal{L}(G(z_G))|
$$

## [Progressive Growing of GANs for Improved Quality, Stability, and Variation(PROGAN-2018)](https://arxiv.org/pdf/1710.10196.pdf)

### 算法原理

<div align=center>
<img src="images/progan_net.png" />
</div>
&emsp;本方法为从低分辨率图像逐渐生成高分辨图像。训练方法: 先设置好训练深度，每个深度输出分辨率是前一个深度的两倍，初始产生4x4的图像，逐渐向后加一个模块(提升分辨率为两倍)，一个深度训练到收敛才训练下一级网络。  

**平滑分辨率**
<div align=center>
<img src="images/progan_smooth.png" />
</div>

