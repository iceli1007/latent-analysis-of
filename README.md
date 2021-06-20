# latent analysis of GANs

[Interpreting the Latent Space of GANs via Measuring Decoupling](https://ieeexplore.ieee.org/document/9399843)
## Abstract 
With the success of generative adversarial networks (GANs) on various real-world applications, the controllability and security of GANs have raised more and more concerns from the community. Specifically, understanding the latent space of GANs, i.e., obtaining the completely decoupled latent space, is essential for applications in some secure scenarios. At present, there is no quantitative method to measure the decoupling of latent space, which is not conducive to the development of the community. In this article, we propose two methods to measure the sensitivity of latent dimensions: one is a sequential intervention method, and the other is an optimization-based method that measures the sensitivity in both the value and the direction. With these two methods, the decoupling of latent space can be measured by the sparsity of the sensitivity vector obtained. The effectiveness of the proposed methods has been verified by experiments on the representative GANs.

-----

## Steps
The motivation of this paper is interpreting the latent space of pre-trained GANs. In order to measure the decoupling of GANs, you should have following steps:

1. Create the latent vector of the real images. You can use any [GAN inversion](https://arxiv.org/abs/2101.05278) technique to achieve it. (In this paper, we use the optimization-based GAN inversion and the latent space of GANs is visualized by TSNE.). The code are available at **create_latent.py** and **create_TSNE.py**

2. Train a classifier to use as the concept metric. You can use any existing framework, such as ResNet.

3. Create the weights of different latent dimensions by the optimization method. The code are available at **create_weight.py** 
