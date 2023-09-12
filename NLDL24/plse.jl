using Revise
using MLDatasets, Knet
using IterTools: ncycle, takenth, takewhile
using KnetNLPModels, NLPModels
# using PartitionedKnetNLPModels
include("src/PartitionedKnetNLPModels.jl")
# include("NLDL24/script_results.jl")
const PK = PartitionedKnetNLPModels

create_minibatch = KnetNLPModels.create_minibatch
printing = true
α = 0.
# max_time = Inf
max_time = 32400.
max_time = 8*3600.
ϵ = 1e-9

#=
MNIST
=#
(xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
data_train = (xtrn, ytrn)
data_test = (xtst, ytst)

C = 10
layer_PS = [24,15,1] 

# max_iter = 30000
max_iter = 60000
size_minibatch = 100
Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
#46740
pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train, data_test, size_minibatch)
ges_plse100 = PK.PLS(pknet_nlp_plse; max_time, max_iter, printing, α, ϵ, linesearch_option=:basic)

io = open("src/optim/results/linesearch_plse.jl", "r")	
s = read(io, String)
io2 = open("src/optim/results/plse_MNIST_basic_1.jl", "w")
write(io2, s)
close(io)
close(io2)

# max_iter = 250000
# size_minibatch = 20
# Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
# #46740
# pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train, data_test, size_minibatch)
# # ges_plse100 = PK.PLS(pknet_nlp_plse; x = x0_minbatch100, max_time, printing, α)
# ges_plse100 = PK.PLS(pknet_nlp_plse; max_time, max_iter, printing, α)

# io = open("src/optim/results/linesearch_plse.jl", "r")	
# s = read(io, String)
# io2 = open("src/optim/results/plse_MNIST_20.jl", "w")	
# write(io2, s)
# close(io)
# close(io2)


#=
CIFAR10
=#

using Revise
using MLDatasets, Knet
using IterTools: ncycle, takenth, takewhile
using KnetNLPModels, NLPModels

# using PartitionedKnetNLPModels
include("src/PartitionedKnetNLPModels.jl")
const PK = PartitionedKnetNLPModels

create_minibatch = KnetNLPModels.create_minibatch
printing = true
α = 0.
max_time = Inf
# max_time = 8*3600.
ϵ = 1e-9

(xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
data_train = (xtrn, ytrn)
data_test = (xtst, ytst)

C = 10
layer_PS = [35,15,1] 

size_minibatch = 100
max_iter = 25000
# max_iter = 50000

Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train, data_test, size_minibatch)
# ges_plse100 = PK.PLS(pknet_nlp_plse; max_time, max_iter, printing, α, ϵ, linesearch_option=:backtracking)
ges_plse100 = PK.PLS(pknet_nlp_plse; max_time, max_iter, printing, α, ϵ, linesearch_option=:basic)

io = open("src/optim/results/linesearch_plse.jl", "r")	
s = read(io, String)
io2 = open("src/optim/results/plse_CIFAR10_100_basic_50epochs.jl", "w")	
write(io2, s)
close(io)   
close(io2)
