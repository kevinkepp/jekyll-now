---
layout: draft
title: Bayesian Neural Networks
---

The purpose of this post is to outline the connection between Neural Networks and Bayesian methods and give an overview of current publications in this domain.

## I. Background

[...]

[TODO why combine Bayesian methods and DNNs?]


## II. Overview of Bayesian Methods for Training NNs

[TODO intro from [Li et al.(2015)](#s27)]

[Welling and Teh (2011)](#s26) introduce Stochastic Gradient Langevin Dynamics, a Stochastic Gradient Markov Chain Monte Carlo (
SG-MCMC) method which combines of Stochastic Gradient Descent (SGD) and Langevin Dynamics.
Adding Gaussian noise with the right variance allows the optimization process to converge to the full posterior over the network parameters instead of just to the mode, i.e. the maximum a posteriori (MAP) estimate, as SGD would do.
Thus, the parameter uncertainty is retained and the network is less likely to overfit the data.
They apply the algorithm to logistic regression, ICA, etc.

However, as [Li et al.(2015)](#s27)] describe, this approach is inefficient when used to train DNNs because of the pathological curvature and saddle points.
This can be tackled with preconditioning methods such as including local geometry, i.e. second-order information such as the expected Fisher information, but usually do not scale well enough for training DNNs.
[Li et al.(2015)](#s27)] introduce preconditioned SGLD (pSGLD) which efficiently preconditions SGLD [TODO how?].

## III. Sources

<a name="s26"></a>Welling, M. and Teh, Y. W. (2011). **Bayesian Learning via Stochastic Gradient Langevin Dynamics**. _In Proceedings of the 28th International Conference on International Conference on Machine Learning, pp. 681–688_.

<a name="s27"></a>Li, C., Chen, C., Carlson, D., & Carin, L. (2015). **Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks**. _[arXiv:1512.07666](http://arxiv.org/abs/1512.07666)_.

