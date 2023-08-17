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


function adam_train_chain(chain; iter_max=5, name="undefined")
  accuracy = []
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, " initial accuracy: ", acc)
  for i in 1:iter_max
    progress!(adam(chain, ncycle(dtrn,1)))
    acc = Knet.accuracy(chain; data=dtst)    
    push!(accuracy, acc)
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
    
  end  
  return accuracy
end


printing = true

C = 10
layer_PS = [24,15,1] 

PSNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
PSNet_NLL = PK.Chain_NLL(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
LeNet_NLL = PK.Chain_NLL(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))
LeNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,1,6; pool_option=1), PK.Conv(5,5,6,16; pool_option=1), PK.Dense(256,120), PK.Dense(120,84), PK.Dense(84,10,identity))


iter_max = 2

one_adam_run!(PSNet_PSLDP, dtrn, dtst; iter_max, name="PSNet_PSLDP")
acc_Adam_LeNet_NLL = adam_train_chain(LeNet_NLL; iter_max, name="LeNet")
acc_Adam_LeNet_PSLDP = adam_train_chain(LeNet_PSLDP; iter_max, name="LeNet_PSLDP") # up to 97-98%

acc_Adam_PSNet_NLL = adam_train_chain(PSNet_NLL; iter_max, name="PSNet_NLL")
acc_Adam_PSNet_PSLDP = adam_train_chain(PSNet_PSLDP; iter_max, name="PSNet_PSLDP")



w_trained_psnet = PartitionedKnetNLPModels.vector_params(PSNet) #46740
# w_trained_lenet = PartitionedKnetNLPModels.vector_params(LeNet) #44426 


# Part_PSNet = PK.PartChainPSLDP(PK.Conv(4,4,3,10; pool_option=1), PK.Conv(4,4,10,10; pool_option=1), PK.SL(250,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity)) # CIFAR
# Part_PSNet = PK.PartChainPSLDP(PK.Conv(4,4,1,10; pool_option=1), PK.Conv(4,4,10,10; pool_option=1), PK.SL(160,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
# Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
# 38360
Part_PSNet = PK.PartChainPSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
#46740

pknet_nlp_plse = PK.PartitionedKnetNLPModel(Part_PSNet; name=:plse, data_train = (xtrn, ytrn), data_test = (xtst, ytst), size_minibatch)

NLPModels.obj(pknet_nlp_plse, pknet_nlp_plse.w) 

# α=0.005 # mean = 0, max =
α=0.
max_time = 300.
max_time = 1800.
max_time = 7200.
# max_time = 10800.

x0 = copy(w_trained_psnet)

ges_plse = PK.PLS(pknet_nlp_plse; x = x0, max_time, printing, α)

# LS
ges_plse = PK.PLS(pknet_nlp_plse; max_time, printing, α)
# 11-12 epochs 91.3%