using MLDatasets
using IterTools: ncycle, takenth, takewhile
using StatsBase

using Knet
using KnetNLPModels
import Base.size

include("src/PartitionedKnetNLPModels.jl")
# include("NLDL24/script_results_adam.jl")
const PK = PartitionedKnetNLPModels

create_minibatch = KnetNLPModels.create_minibatch

iter_max = 100
max_seed = 10
size_minibatch = 100

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

Adam_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "MNIST")

#= 
CIFAR10
=#
(xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

C = 10 # number of classes
layer_PS = [35,15,1] 

LeNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,3,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(400,200), PK.Dense(200,100), PK.Dense(100,10,identity))
LeNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,3,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(400,200), PK.Dense(200,100), PK.Dense(100,10,identity))
PSNet_NLL = () -> PK.Chain_NLL(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))

Adam_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "CIFAR10")

# printing graphs
archicture_names = String["LeNet", "PSNet"]
loss_names = String["NLL", "PSLDP"]
# minibatch_sizes = Int[20, 100]
minibatch_sizes = Int[100]
dataset_names = ["CIFAR10", "MNIST"]
all_graphs(archicture_names, loss_names, minibatch_sizes, dataset_names, debug=true)



#=
Test hyper-parameters
=#

lrs = [0.001, 0.002, 0.0001]
beta1s = [0.8, 0.9, 0.95]
beta2s = [0.99, 0.999, 0.9999]
iter_max = 50
size_minibatch = 600
max_seed = 5

(xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

C = 10 # number of classes
layer_PS = [35,15,1] 
PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))

Adam_test_hyper_parameters(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed; name_dataset = "CIFAR10", lrs, beta1s, beta2s)

all_hyperparameters(;lrs, beta1s, beta2s)