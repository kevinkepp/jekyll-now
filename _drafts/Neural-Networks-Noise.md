---
layout: draft
title: Neural Networks and Noise
sitemap: false
---

The purpose of this post is to understand the role noise plays in training [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) using stochastic optimization methods.

[...]

## Optimization

When training neural networks we define a loss function $$ L(X,\theta) $$ that measures the loss of the network with weights $$ \theta $$ with respect to the data set $$ X $$.
For example we could use the mean squared error
\begin{equation} 
L(X,\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_{i})^2
\end{equation}
where $$ \hat{y}_{i} = f(x_i) $$ is the prediction of the network.
To find good parameter values we initialize $$ \theta $$ randomly and then use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to minimize the loss function by following the gradient $$ \nabla L(X,\theta) $$ and updating $$ \theta $$ accordingly.

As long as the gradient is non-zero and we choose a small enough step size, gradient descent is guaranteed to make *local progress*, i.e. moves our weights closer to a local optimum.
When the gradient is zero we reached a **critical point** and the algorithm gets stuck \[[1](#s1)\]. 
For (strongly) convex problems this point always represents the global optimum.
However, for neural networks the optimization landscape is usually highly complex, i.e. non-convex.
Thus, the critical point could be a either a local minimum or a saddle point.

There are (at least) two problems here.
First, how do we know our local minimum is actually the global one?
And second, when we hit a saddle point, how do we escape it?

### Adding Noise

Both problems can be solved using a noisy gradient signal, for example by applying Stochastic Gradient Descent (SGD).
Here, parameter updates are computed based on stochastic gradients $$ \tilde{g} $$ that are computed using randomly selected batches of training samples and thus
\begin{equation}
\mathbf{E}[\tilde{g}(\theta)] = \nabla L(\theta).
\end{equation}

The noisy gradient helps to escape saddle points while not interfering with the convergence in convex regimes \[[2](#s2)\].
We could also use second-order information, i.e. the Hessian, to detect a saddle point, this however is computationally very expensive.
There is more recent research that shows that first-order methods in general actually only very rarely converge to saddle points \[[3](#s3), [4](#s4)\].
[TODO Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. ArXiv:1406.2572 [Cs, Math, Stat]. Retrieved from http://arxiv.org/abs/1406.2572]

Regarding the optimality of local minima, it was shown that for large networks, all local minima found by SGD are in fact local minima of "high quality measured by the test error" \[[5](#s5)\].
It is thus not necessarily required to find the global minimum.

In fact, as our actual goal in machine learning is not to minimize the empirical risk but the generalization error, the global optimum might very well be a solution that overfits the training data and results in lower generalization performance, i.e. higher error on the test set [REF?].
This is also true for sharp local minima.
Ideally, we want to end up in a minimum with a *flat* regime that can be robustly found by the optimization procedure.

### Adding More Noise

Such a *flat minimum* can be found by adding more noise to the gradient during optimization.

From a probabilistic perspective minimizing the loss function using SGD is equivalent to a maximum a posteriori (MAP) estimation of the network parameters where the likelihood represents the cost function and the prior represents the parameter regularization.
Such a MAP estimation converges to the mode of the posterior distribution of the network parameters, i.e. the most likely parameter configuration.
Doing this, we loose the remaining parameter uncertainty and thus may overfit the training data.

Alternatively, we could use Markov chain Monte Carlo (MCMC) methods to sample from the posterior distribution of network parameters.
With such methods, the parameter uncertainty is retained because they consider the whole posterior distribution and not just the mode.
The problem with MCMC methods is the high computational cost of considering the full dataset in every iteration.

Stochastic Gradient Langevin Dynamics (SGLD) combines SGD with Langevin dynamics, a MCMC technique \[[6](#s6)\].
SGLD uses the stochastic optimization update rule but adds Gaussian noise to each parameter update.
As a result, we can use batches of training samples but at the same time sample from the full posterior distribution, i.e. retaining uncertainty in the network parameters, instead of only converging to a mode.

[TODO] How does retaining parameter uncertainty relate to the flat minimum regime we want to converge to?

[TODO] Why is this not applied to NNs in practice?

### The Downside

As we have discussed so far, adding noise helps to guide our optimization procedure to find solution with better generalization performance.
However, the downside of noisy parameter gradients is that they slow down the procedure.

[TODO] Relate to SVRG, SAG, etc. which have been designed to reduce variance of the stochastic gradients to speed up convergence.

[TODO] How to do a trade-off?


## Sources
<a name="s1"></a>[1] [http://www.offconvex.org/2016/03/22/saddlepoints/](http://www.offconvex.org/2016/03/22/saddlepoints/).

<a name="s2"></a>[2] Ge R., Huang F., Jin C. and Yuan Y. (2015). **Escaping From Saddle Points - Online Stochastic Gradient for Tensor Decomposition**. [_arXiv:1503.02101_](https://arxiv.org/abs/1503.02101).

<a name="s3"></a>[3] Lee D. J., Simchowitz M., Jordan I. M. and Recht B. (2016). **Gradient Descent Converges to Minimizers**. [_arXiv:1602.04915_](https://arxiv.org/abs/1602.04915).

<a name="s4"></a>[4] Lee D. J., Panageas I., Piliouras G., Simchowitz M., Jordan I. M. and Recht B. (2017). **First-order Methods Almost Always Avoid Saddle Points**. [_arXiv:1710.07406_](https://arxiv.org/abs/1710.07406).

<a name="s5"></a>[5] Choromanska A., Henaff M., Mathieu M., Arous G. B and LeCun Y. (2014). **The Loss Surfaces of Multilayer Networks**. [_arXiv:1412.0233_](https://arxiv.org/abs/1412.0233).

<a name="s6"></a>[6] Welling M. and Teh Y. W. (2011). **Bayesian Learning via Stochastic Gradient Langevin Dynamics**. _In Proceedings of the 28th International Conference on International Conference on Machine Learning Pp. 681–688. ICML’11._
