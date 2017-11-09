---
layout: draft
title: Compressed Domain Learning
sitemap: false
---

The purpose of this post is to give an overview of different methods on how to train neural networks more efficiently.
The goal of these techniques is to reduce resources needed to train neural networks, for example, in order to use them in resource-constraint environments like on embedded devices, but also to make training generally more efficient and thus faster.
We focus mainly on compression techniques such as quantization of weights, activations and/or gradients during training.

## I. Background

### Quantization

[TODO...]
[TODO discretization]

### Reduced Precision

[TODO...]

### Fixed-point Format

[Fixed-point numbers](https://en.wikipedia.org/wiki/Fixed-point_arithmetic) are fractional numbers represented by an exponent, or scaling factor, and an underlying integer number which is to be scaled, also called mantissa.
For example, the value $$ 1.23 $$ can be represented as mantissa $$ 1230 $$ and scaling factor $$ 1/1000 $$.

For computations, the scaling factor is usually a power of 2.
In this case, the scaling factor represents the position of the point in the binary representation of the number as described by [Hayden So (2006)](#s3).
This binary point splits the number in an integer and a fractional part.
For example in a fixed-point format $$ Q5.2 $$ we have 5 bits for the integer part, 2 bits for the fractional part and 1 sign bit.
Here, the number $$ 010110.11_b $$ corresponds to $$ 2^4 + 2^2 + 2^1 + 2^{-1} + 2^{-2} = 22.75 $$ and $$ 100001.01_b $$ corresponds to $$ -2^5 + 2^1 + 2^{-2} = -30.75 $$.

Usually fixed-point numbers of the same type share the same (fixed) scaling factor and thus only need to store the fractional part.
Often, the bit width of the fractional part is also referred to as precision.

[TODO relation to quantization and precision]


## II. Overview Quantization for DNNs

[Hashemi et al. (2016)](#s24) give an overview of the impact of precision quantization on the accuracy and energy of neural networks.
[TODO extend]

### Fixed-point Format

[Courbariaux et al. (2014)](#s7) evaluate floating-point, fixed-point and dynamic fixed-point formats (for all numeric values in the network) with different precisions.
In the dynamic fixed-point format, each layer's weights, bias, weighted sums and (post-linearity) outputs have their own scaling factor.
They found that half-precision floating point has almost no impact on the performance, for fixed-point format a bit width of 20 is sufficient, whereas for dynamic fixed-point a bit width of 10 for the forward/backward propagation and 12 for the parameter updates is sufficient.

[Gutpa et al. (2015)](#s1) show that 16 bit fixed-point format parameters, activations (layer outputs), back-propagated errors and parameters updates can be sufficient for training DNNs and CNNs on MNIST and CIFAR datasets (with problem-specific precision).

### Very Low Precision

[Soudry et al. (2014)](#s8) train neural networks with Expectation Propagation, a variational Bayes method, to approximate the distribution of the weights.
This allows for parameter-free training and for discretization of the resulting weights.
They achieve SOTA results with binarized weights on binary text classification tasks.
However, binary values are only used during inference.
[TODO verify]

[Courbariaux et al. (2015)](#s9) extend the idea behind [(Soudry et al., 2014)](#s8) and achieve near SOTA performance using CNNs on several datasets.
It is called Binary Connect.
[TODO extend]

[Lin et al. (2015)](#s11) extend Binary Connect [(Courbariaux et al., 2015)](#s9) to Ternary Connect [...].
[TODO full-precision weights during inference and quantize the neurons only during the back propagation process, and not during forward propagation?]
[TODO extend]

[Courbariaux et al. (2016)](#s4) and [Hubara et al. (2016)](#s17) binarize weights and activations using a (mostly) deterministic sign function as an activation function in each layer.
Only activations during train-time are rounded stochastically.
The resulting network is called Binarized Neural Network (BNN).
Binary values allow for replacing most arithmetic operations with bit-wise operations, e.g. Xnor-counts for dot products.
During the backward pass, the parameter gradients are not binarized but kept as regular real-valued numbers which they claim is necessary for SGD to small noisy parameter steps.
They use an adapted derivative ("straight-through estimator") when propagating through the sign non-linearity, otherwise gradients would be 0 mostly as the derivative of the sign function is two times the [Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function).
Weights are stored and adapted as real values but clipped to stay in [-1,1] and binarized when used.
Shift-based implementations of batch norm and ADAM are used.
They achieve almost SOTA accuracy on MNIST, CIFAR-10, SVHN and using binary-optimized GPU kernels they achieve a 7x run-time speedup (due to memory size and access reduction) on MNIST without accuracy loss.

[Zhou et al. (2016)](#s15) generalize [(Courbariaux et al., 2016)](#s4) and show an AlexNet with 1-bit weights and 2-bit activations, called DoReFa-Net, that can be trained from scratch using 6 bit gradients and achieve comparable accuracy.
They use deterministic quantization for weights and activations but stochastic quantization for gradients.
Efficient bit convolution kernels based on Xnor and count operations exploit low bit-width fixed-point numbers.

[Chen et al. (2017)](#s2) binarize weights and activations during forward pass and use low-resolution fixed-point gradients during backward pass.
The resulting network is called FxpNet.
They introduce Integer Batch Normalization and fixed-point ADAM.
[TODO extend]

[Narodytska et al. (2017)](#s25) verify the properties of binarized NNs...
[TODO extend]

### Distributed Learning

[Seide et al. (2014)](#s12) quantize the gradients before distributing to the data-parallel peers and synchronuously update the parameters.
The quantization error is kept and applied to the next minibatch before quantization.
The quantization uses 1 bit and a threshold of 0 whereas the unquantizer uses two values per weight-matrix column such that the reconstruction error is minimized.
Achieves speedup of 4-6x using 8 GPUs over using a single one without parallelization.

[Alistarh et al. (2016)](#s5) use a custom stochastic (sparsifying) quantization scheme (QSGD) when distributing the (stochastic) gradients to peer processors.
All processors then synchronously aggregate the gradients distributed by the peers, decode them and apply the update to their local parameter copy $$ x $$.
The quantization schema is controlled by a tuning parameter $$ s $$ which determines the code length (between $$ O(\sqrt{n}) $$ and $$ O(n) $$) and the (bounded) noise added.
They roughly speedup the epoch time by 2x using 4-bit gradient quantization, largely due to a big decrease in communication time, on ImageNet, CIFAR-10, AN4.
Furthermore, they provide global/local convergence guarantees for convex/non-convex problems.

[Wen et al. (2017)](#s6) use a similar (or the same?) quantization scheme as [(Alistarh et al., 2016)](#s5).
[TODO called TernGrad]
[TODO extend]

[Lian at al. (2017)](#s19) [TODO...]

### Other

[Han et al. (2015)](#s10) train the network to learn which connections are important, prune the unimportant connections and then fine-tune the weights of the remaining connections.
[TODO extend]

[Esser et al. (2015)](#s20) [TODO...]

[Chen et al. (2015)](#s13) enforce random weight sharing, depending on memory constraints, using a hash map.
They exploit the redundancy in neural network parameters as described for example by [(Denil et al., 2013)](#s14).
[TODO results show largely improved generalization?!]

[Miyashita et al. (2016)](#s21) use logarithmic weight representation instead of fixed-point format to encode weights using 3 bits without loss in accuracy.
They also introduce 5 bit log representation training which is better than 5 bit linear representation.
[TODO only CNNs?]
[TODO extend]

[Micikevicius et al. (2017)](#s16) employ a mixed-precision training where they use half precision floating-point when computing activations and gradients and updating weights but store weights in single precision.
Accumulation during MAC operations also has to be done in single precision.
This mixed precision idea is used by the latest generation of NVIDIA GPUs ("Volta").
The memory consumption can be reduced by ~2x.


### Not Categorized Yet [TODO...]

[Chen et all. (2017)](#s22)

[Sun et al. (2017)](#s23)

[Anonymous (2017)](#s18)


## III. Sources
<a name="s1"></a>Gupta, S., Agrawal, A., Gopalakrishnan, K. and Narayanan, P. (2015). **Deep Learning with Limited Numerical Precision**. [_arXiv:1502.02551_](https://arxiv.org/abs/1502.02551).

<a name="s2"></a>Chen, X., X. Hu, H. Zhou, and N. Xu (2017). **FxpNet: Training a Deep Convolutional Neural Network in Fixed-Point Representation**. _In 2017 International Joint Conference on Neural Networks (IJCNN), Pp. 2494–2501_.

<a name="s3"></a>Hayden So (2006). **Introduction to Fixed Point Number Representation**. _[University of California Berkely, CS61c Spring 2006](http://www-inst.eecs.berkeley.edu/~cs61c/sp06/handout/fixedpt.html)_.

<a name="s4"></a>Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R. and Bengio, Y. (2016). **Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.** [_arXiv:1602.02830_](http://arxiv.org/abs/1602.02830).

<a name="s5"></a>Alistarh, D., Grubic, D., Li, J., Tomioka, R. and Vojnovic, M. (2016). **QSGD: Communication-Optimal Stochastic Gradient Descent, with Applications to Training Neural Networks**. [_arXiv:1610.02132_](http://arxiv.org/abs/1610.02132).

<a name="s6"></a>Wen, W., Xu, C., Yan, F., Wu, C., Wang, Y. and Chen, Y. (2017). **TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning**. [_arXiv:1705.07878_](http://arxiv.org/abs/1705.07878).

<a name="s7"></a>Courbariaux, M., Bengio, Y., and David, J. (2014). **Training Deep Neural Networks with Low Precision Multiplications**. [_arXiv:1412.7024_](http://arxiv.org/abs/1412.7024).

<a name="s8"></a>Soudry, D., Hubara, I., and Meir, R. (2014). **Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights**. _In Advances in Neural Information Processing Systems 27, Pp. 963–971_.

<a name="s9"></a>Courbariaux, M., Bengio, Y. and Jean-Pierre, David (2015). **BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations**. _In Advances in Neural Information Processing Systems 28, Pp. 3123–3131_.

<a name="s10"></a>Han, S., Pool, J., Tran, J. and Dally, W. (2015). **Learning Both Weights and Connections for Efficient Neural Network**. _In Advances in Neural Information Processing Systems 28, Pp. 1135–1143_.

<a name="s11"></a>Lin, Z., Courbariaux, M., Memisevic, R. and Bengio, Y. (2015). **Neural Networks with Few Multiplications**. [_arXiv:1510.03009_](http://arxiv.org/abs/1510.03009).

<a name="s12"></a>Seide, F., Fu, H., Droppo, J., Li, G., & Yu, D. (2014). **1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs**. _Microsoft Research_.

<a name="s13"></a>Chen, W., Wilson, J., Tyree, S., Weinberger, K., and Chen, Y. (2015). **Compressing neural networks with the hashing trick**. _In International Conference on Machine Learning, pp. 2285–2294_.

<a name="s14"></a>Denil, M., Shakibi, B., Dinh, L., Ranzato, M. A., and de Freitas, N. (2013). **Predicting Parameters in Deep Learning**. _In Advances in Neural Information Processing Systems 26, pp. 2148–2156_.

<a name="s15"></a>Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., and Zou, Y. (2016). **DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients**. _[arXiv:1606.06160](http://arxiv.org/abs/1606.06160)_.

<a name="s16"></a>Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G. and Wu, H. (2017). **Mixed Precision Training**. _[arXiv:1710.03740](http://arxiv.org/abs/1710.03740)_.

<a name="s17"></a>Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., and Bengio, Y. (2016). **Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations**. _[arXiv:1609.07061](http://arxiv.org/abs/1609.07061)_.

<a name="s18"></a>Anonymous (2017). **Training and Inference with Integers in Deep Neural Networks**. _[OpenReview, ICLR 2018 Conference Blind Submission](https://openreview.net/forum?id=HJGXzmspb&noteId=HJGXzmspb)_.

<a name="s19"></a>Lian et al... (2017). **Asynchronous Decentralized Parallel Stochastic Gradient Descent**.

<a name="s20"></a>Esser et al... (2015). **Backpropagation for Energy-Efficient Neuromorphic Computing**.

<a name="s21"></a>Miyashita et al... (2016). **Convolutional Neural Networks using Logarithmic Data Representation**.

<a name="s22"></a>Chen et al... (2017). **Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks**.

<a name="s23"></a>Sun et al... (2017) **meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting**

<a name="s24"></a>Hashemi et al... (2016). **Understanding the Impact of Precision Quantization on the Accuracy and Energy of Neural Networks**.

<a name="s25"></a>Narodytska et al... (2017). **Verifying Properties of Binarized Deep Neural Networks**.
