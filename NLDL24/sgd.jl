using MLDatasets
using IterTools: ncycle, takenth, takewhile
using StatsBase

using Knet
using KnetNLPModels
import Base.size

include("src/PartitionedKnetNLPModels.jl")
include("NLDL24/script_results_sgd.jl")
const PK = PartitionedKnetNLPModels

create_minibatch = KnetNLPModels.create_minibatch

iter_max = 100
max_seed = 5

#= 
MNIST
=#
(xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

C = 10 # number of classes
layer_PS = [24,15,1] # individual score neurons composing the successive searable layers
# in total, it contains respectively : 240, 150 and 10 neurons (e.g. layer_Ps .* C).

LeNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))
LeNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))
PSNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# LeNet 44426 # length(vector_params(LeNet_NLL()))
# PSNet 53780 # length(vector_params(PSNet_PSLDP()))
# every raw score depends only on 6340 weights # length(PS(PSNet_PSLDP())[1])
# every element function depends on 11588 weights # length(build_listes_indices(PSNet_PSLDP())[1,2]) (or [i,j] with 1 ≤ i != j ≤ C)
# 1092 common weights for every score

# size_minibatch = 20
# SGD_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "MNIST")

size_minibatch = 100
SGD_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "MNIST")


#= 
CIFAR10
=#
(xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

C = 10 # number of classes
layer_PS = [35,15,1] 

element_function_indices[1,1]
LeNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,3,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(400,200), PK.Dense(200,100), PK.Dense(100,10,identity))
LeNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,3,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(400,200), PK.Dense(200,100), PK.Dense(100,10,identity))
PSNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# LeNet 103882 # length(vector_params(LeNet_NLL()))
# PSNet 81750 # length(vector_params(PSNet_PSLDP()))
# every raw score depends only on 12279 weights # length(PS(PSNet_PSLDP())[1])
# every element function depends on 19998 weights # length(build_listes_indices(PSNet_PSLDP())[1,2]) (or [i,j] with 1 ≤ i != j ≤ C)
# 4560 common weights for every score

# size_minibatch = 20
# SGD_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "CIFAR10")

size_minibatch = 100
SGD_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "CIFAR10")

# weights = vector_params(LeNet_NLL) # return the weights as a (Cu)Vector
# size(weights) #  return the size of the neural network

archicture_names = String["LeNet", "PSNet"]
loss_names = String["NLL", "PSLDP"]
# minibatch_sizes = Int[20, 100]
minibatch_sizes = Int[100]
dataset_names = ["CIFAR10", "MNIST"]
all_graphs(archicture_names, loss_names, minibatch_sizes, dataset_names, debug=true)



# (xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
# (xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

# size_minibatch = 100
# dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
# dtst = create_minibatch(xtst, ytst, size_minibatch)

# size_minibatch = 600
# dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
# dtst = create_minibatch(xtst, ytst, size_minibatch)


# LeNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))
# LeNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))
# PSNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))

# iter_max = 100

# acc_sgd = one_sgd_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0, lr=0.0025)
# acc_sgd = one_sgd_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0)
# acc_nesterov = one_nesterov_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0)
# acc_momentum = one_momentum_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0)

# acc_adagrad = one_adagrad_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0) #--
# acc_adadelta = one_adadelta_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0)
# acc_rmsprop = one_rmsprop_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0)
# acc_adam = one_adam_run!(LeNet_PSLDP(), dtrn, dtst; iter_max, name="Test", index_seed=0)

# acc_sgd = one_sgd_run!(LeNet_NLL(), dtrn, dtst; iter_max, name="Test", index_seed=0)

# using NLPModels

# LeNet_PSLDP_ = LeNet_PSLDP()
# data_train = (xtrn, ytrn)
# data_test = (xtst, ytst)
# knet_nlp = KnetNLPModel(LeNet_PSLDP_; data_train, data_test, size_minibatch)

# x = copy(vector_params(knet_nlp))
# NLPModels.obj(knet_nlp, x)
# g = copy(NLPModels.grad(knet_nlp, x))
# x1 = similar(x)
# x2 = similar(x)
# x3 = similar(x)
# x1 .= x .- 0.1 .* g
# x2 .= x .- 0.01 .* g
# x3 .= x .- 0.001 .* g

# NLPModels.obj(knet_nlp, x1)
# g1 = NLPModels.grad(knet_nlp, x1)

# NLPModels.obj(knet_nlp, x2)
# g2 = NLPModels.grad(knet_nlp, x2)

# NLPModels.obj(knet_nlp, x3)
# g3 = NLPModels.grad(knet_nlp, x3)

# set_vars!(knet_nlp, x1)
# KnetNLPModels.accuracy(knet_nlp)
# set_vars!(knet_nlp, x2)
# KnetNLPModels.accuracy(knet_nlp)
# set_vars!(knet_nlp, x3)
# KnetNLPModels.accuracy(knet_nlp)

# knet_nlp.chain(dtrn)

# knet_nlp.w == vector_params(LeNet_PSLDP_)