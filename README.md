# PartitionedKnetNLPModels : A partitioned quasi-Newton stochastic method to train partially separable neural networks

| **Documentation** | **Coverage** | **DOI** |
|:-----------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://paraynaud.github.io/PartitionedKnetNLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://paraynaud.github.io/PartitionedKnetNLPModels.jl/dev
[codecov-img]: https://codecov.io/gh/paraynaud/PartitionedKnetNLPModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/paraynaud/PartitionedKnetNLPModels.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.822073-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.822073

## Motivation
The module address a partially separable loss function, such as the neural network training minimize a partially separable functions
$$
f(x) = \sum_{=1}^N \hat{f}_i (U_i x), \quad f \in \R^n \to \R, \quad \hat f_i:\R^{n_i} \to \R, \quad U_i \in \R^{n_i \times n}.
$$
$f$ is a sum of element functions $\hat{f}_i$, and usually $n_i \ll n$. $U_i$ is a linear operator, it selects the variables used by $\hat{f}_i$.

PartitionedKnetNLPModels.jl define a stochastic trust-region method exploiting the partitioned structure of the derivatives of $f$, the gradient 
$$
\nabla f(x) = \sum_{i=1}^N U_i^\top \nabla \hat{f}_i (U_i x),
$$
and the hessian 
$$
\nabla^2 f(x) = \sum_{i=1}^N U_i^\top \nabla^2 \hat{f_i} (U_i x) U_i,
$$
are the sum of the element derivatives $\nabla \hat{f}_i,  \nabla^2\hat{f}_i$.
This structure allows to define a partitioned quasi-Newton approximation of $\nabla^2 f$
$$
B = \sum_{i=1}^N U_i^\top \hat{B}_{i} U_i,
$$
such that each $\hat{B}_i \approx \nabla^2 \hat{f}_i$.
Contrary to the BFGS and SR1 updates, respectively of rank 1 and 2, the rank of update $B$ is proportionnal to $\min(N,n)$.

#### Reference
* A. Griewank and P. Toint, [*Partitioned variable metric updates for large structured optimization problems*](10.1007/BF01399316), Numerische Mathematik volume, 39, pp. 119--137, 1982.


## Content
PartitionedKnetNLPModels.jl define
- A new layer architecture, called "separable layer".
This layer requires : the size of the previous layer `p` and the next layer `nl` and the number of classes `C`
```julia
separable_layer = SL(p,C,nl/C)
```
- A partially separable loss function PSLDP (partially separable loss determinist prediction)
- A stochastic trust region method which use a partitioned quasi-Newton linear operator to make a quadratic approximate of the PSLDP function

We assume that the reader are familiar with [Knet](https://github.com/denizyuret/Knet.jl) or with neural networks, otherwise [here is the Knet tutorials](https://github.com/denizyuret/Knet.jl/).

First, you have to define the architecture of your neural network. Here the `PSNet` architecture is made of convolution layer `Conv` and from separable layer `SL`
```julia
using PartitionedKnetNLPModels

C = 10 # number of class for MNIST
layer_PS = [40,20,1] 
PSNet = PartChainPSLDP(Conv(4,4,1,20), Conv(4,4,20,50), SL(800,C,layer_PS[1]), SL(C*layer_PS[1],C,layer_PS[2]), SL(C*layer_PS[2],C,layer_PS[3];f=identity))
```

The dataset MNIST
```julia
(xtrn, ytrn) = MNIST.traindata(Float32)
ytrn[ytrn.==0] .= 10
data_train = (xtrn, ytrn) # training dataset

(xtst, ytst) = MNIST.testdata(Float32)
ytst[ytst.==0] .= 10
data_test = (xtst, ytst) # testing dataset
```
Then, define the `PartitionedKnetNLPModel` associated 
```julia
nlp_plbfgs = PartitionedKnetNLPModel(PSNet; name=:plbfgs, data_train, data_test)
```
`nlp_plbfgs` handles: the evaluation of `PSNet` using a minibatch of the `data_train`, the explicit computation of the objective function and its derivatives 
```julia
n = length(nlp_plbfgs.meta.x0) # size of the minimization problem
w = rand(n) # random point 
f = NLPModels.obj(nlp_plbfgs, w) # compute the loss function
g = NLPModels.grad(nlp_plbfgs, w) # compute the gradient of the loss function
```

From these features, PartitionedKnetNLPModels.jl define a stochastic trust region `PUS` (partitioned update solver) using partitioned quasi-Newton update
```julia
PUS(nlp_plbfgs; max_time, max_iter)
```

To use other quasi-Newton approximation than PLBFGS you have to define new `PartitionedKnetNLPModel` with other `name`, similarly to
```julia
nlp_plsr1 = PartitionedKnetNLPModel(PSNet; name=:plsr1, data_train, data_test)
nlp_plse = PartitionedKnetNLPModel(PSNet; name=:plse, data_train, data_test)
```

## Dependencies
The module [Knet](https://github.com/denizyuret/Knet.jl) is used to define the operators required by the neural network such as : convolution, pooling, in a way that neural network can run on a GPU.

[KnetNLPModels](https://github.com/paraynaud/KnetNLPModels.jl) provide an interface between a Knet neural network and the [ADNLPModel](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl).

The partitioned quasi-Newton operators used in the partially separable training are defined in [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl).


## How to install
```
julia> ]
pkg> add https://github.com/paraynaud/PartitionedStructures.jl, https://github.com/paraynaud/KnetNLPModels.jl, https://github.com/paraynaud/PartitionedKnetNLPModels.jl, 
pkg> test PartitionedKnetNLPModels
```
