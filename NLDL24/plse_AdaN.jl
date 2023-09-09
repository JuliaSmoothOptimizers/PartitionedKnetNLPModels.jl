using Revise
using MLDatasets, Knet
using IterTools: ncycle, takenth, takewhile
using KnetNLPModels, NLPModels
using PartitionedKnetNLPModels

include("src/PartitionedKnetNLPModels.jl")
const PK = PartitionedKnetNLPModels

create_minibatch = KnetNLPModels.create_minibatch
printing = true
α = 0.
max_time = Inf
max_time = 32400.
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

max_iter = 30000
size_minibatch = 100
Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))

#46740
pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train, data_test, size_minibatch)
ges_plse_NA100 = PK.PLS_AdaN(pknet_nlp_plse; max_time, max_iter, printing, α, ϵ)

io = open("src/optim/results/linesearch_AdaN_plse.jl", "r")
s = read(io, String)
io2 = open("src/optim/results/plse_AdaN_MNIST_100_new_1.jl", "w")
write(io2, s)
close(io)
close(io2)

