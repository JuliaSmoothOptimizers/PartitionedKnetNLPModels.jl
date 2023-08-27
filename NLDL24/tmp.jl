using Revise
using MLDatasets, Knet
using IterTools: ncycle, takenth, takewhile
using KnetNLPModels, NLPModels
# using PartitionedKnetNLPModels
include("src/PartitionedKnetNLPModels.jl")
include("NLDL24/script_results.jl")
const PK = PartitionedKnetNLPModels


create_minibatch = KnetNLPModels.create_minibatch

(xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
size_minibatch = 20
# size_minibatch = 100
dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
dtst = create_minibatch(xtst, ytst, size_minibatch)
data_train = (xtrn, ytrn)
data_tst = (xtst, ytst)

printing = true

C = 10
layer_PS = [24,15,1] 

PSNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
PSNet_NLL = PK.Chain_NLL(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
LeNet_NLL = PK.Chain_NLL(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))
LeNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))


iter_max = 2

# Knet.accuracy(PSNet_PSLDP,dtst)
accuracy_PSNet_PSLDP = one_adam_run!(PSNet_PSLDP, dtrn, dtst; iter_max, name="PSNet_PSLDP")
w_trained_psnet = PartitionedKnetNLPModels.vector_params(PSNet_PSLDP) #46740

x0_minbatch20 = copy(w_trained_psnet)
x0_minbatch100 = copy(x0_minbatch20)

# # 11-12 epochs 91.3%
α = 0.
max_time = 0.5*3600.
max_time = 4.5*3600.

size_minibatch = 100
data_train = (xtrn, ytrn)
data_test = (xtst, ytst)

Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
#46740
pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train, data_test, size_minibatch)
# ges_plse100 = PK.PLS(pknet_nlp_plse; x = x0_minbatch100, max_time, printing, α)
ges_plse100 = PK.PLS(pknet_nlp_plse; max_time, printing, α)

io = open("src/optim/results/linesearch_plse.jl", "r")	
s = read(io, String)
io2 = open("src/optim/results/linesearch_plse_alone100.jl", "w")	
write(io2, s)
close(io)
close(io2)

size_minibatch = 20
data_train = (xtrn, ytrn)
data_test = (xtst, ytst)

Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
#46740
pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train, data_test, size_minibatch)
ges_plse20 = PK.PLS(pknet_nlp_plse; x = x0_minbatch20, max_time, printing, α)

io = open("src/optim/results/linesearch_plse.jl", "r")	
s = read(io, String)
io2 = open("src/optim/results/linesearch_plse20.jl", "w")	
write(io2, s)
close(io)
close(io2)

PSNet_PSLDP_ = PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
psnet_nlp = KnetNLPModel(PSNet_PSLDP_; data_train, data_test)

size_minibatch=100
Part_PSNet_ = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
pknet_nlp_plse_ = PK.PartitionedKnetNLPModel(Part_PSNet_; name=:plse, data_train, data_test, size_minibatch)
ges_lbfgs = PK.LBFGS(pknet_nlp_plse_; max_time, printing)

io = open("src/optim/results/accuracy_LBFGS.jl", "r")	
s = read(io, String)
io2 = open("src/optim/results/lbfgs100.jl", "w")	
write(io2, s)
close(io)
close(io2)
