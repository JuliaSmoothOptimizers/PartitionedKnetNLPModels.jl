# tmux a -t LBFGS
using Revise
using MLDatasets, Knet
using IterTools: ncycle, takenth, takewhile
using KnetNLPModels, NLPModels
using JSOSolvers
using StatsBase
# using PartitionedKnetNLPModels
include("src/PartitionedKnetNLPModels.jl")
include("NLDL24/script_results.jl")
const PK = PartitionedKnetNLPModels

create_minibatch = KnetNLPModels.create_minibatch

function seeded_lbfgs(model, data_train, data_test;  
  max_seed=10,  
  size_minibatch=100,
  dataset_name = "MNIST",
  max_time=30.,
  max_epoch=100,
  mem=5)
  
  knet_nlp = KnetNLPModel(model(); data_train, data_test, size_minibatch)
  iter_epoch = length(knet_nlp.training_minibatch_iterator)
  max_iter = iter_epoch * max_epoch  
  accuracies = zeros(Float64, max_epoch, max_seed)

  for seed in 1:max_seed
    println("LBFGS-", mem, ", ", seed, "-th seed, max_iter : ", max_iter, " max_time: ", max_time)

    accuracy = Float32[]
    function callback(nlp, solver, stats)
      KnetNLPModels.minibatch_next_train!(nlp)
      if (stats.iter % iter_epoch == 0) && (stats.iter != 0)
        acc = KnetNLPModels.accuracy(nlp)
        println("time : ", stats.elapsed_time, ", iter : ", stats.iter, ", accuracy: ", acc)
        push!(accuracy, acc)
      end
    end    
    
    knet_nlp = KnetNLPModel(model(); data_train, data_test, size_minibatch)        
        
    solver = JSOSolvers.LBFGSSolver(knet_nlp; mem)
    ges = solve!(solver, knet_nlp; max_time, callback, max_iter, verbose=0)
  
    # ges = JSOSolvers.lbfgs(knet_nlp; max_time, callback, max_iter, verbose=0)

    view_acc = view(accuracies, 1:max_epoch, seed)
    view_acc[1:length(accuracy)] .= accuracy
  end

  (_mean, _std) = mean_and_std(accuracies, 2)
  mean = Vector(_mean[:,1])
  std = Vector(_std[:,1])

  println("mean % LBFGS : \n", mean)
  println("std % LBFGS : \n", std)  
  io = open("src/optim/results/lbfgs_$(dataset_name)_$(size_minibatch)_mem$(mem).jl", "w")    
  print(io, "accuracies_LBFGS = ", accuracies, "\n\n")
  print(io, "mean_LBFGS = ", mean, "\n\n")
  print(io, "std_LBFGS = ", std, "\n\n")
  close(io)

  return mean, std
end

function memory_test_lbfgs(model, data_train, data_test;  
                  mems=[5], kwargs...)
  for mem in mems
    seeded_lbfgs(model, data_train, data_test; mem, kwargs...)
  end
end


max_time = Inf
size_minibatch=100
max_seed=10
max_epoch=100
mems = [2,5,10,15]
#=
MNIST
=#

# max_time = Inf
# size_minibatch=600
# max_seed=2
# max_epoch=3

(xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
data_train = (xtrn, ytrn)
data_test = (xtst, ytst)

C = 10 # number of classes
layer_PS = [24,15,1] # individual score neurons composing the successive searable layers

PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))

memory_test_lbfgs(PSNet_PSLDP, data_train, data_test; max_seed, max_time, size_minibatch, max_epoch, dataset_name = "MNIST", mems)

#=
CIFAR10
=#
(xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
data_train = (xtrn, ytrn)
data_test = (xtst, ytst)

C = 10 # number of classes
layer_PS = [35,15,1] 
PSNet_PSLDP = () -> PK.Chain_PSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))

memory_test_lbfgs(PSNet_PSLDP, data_train, data_test; max_seed, size_minibatch, max_time, max_epoch, dataset_name = "CIFAR10", mems)


# accuracy = Float32[]
# size_minibatch=100
# PSNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
# knet_nlp = KnetNLPModel(PSNet_PSLDP; data_train, data_test, size_minibatch)
# JSOSolvers.lbfgs(knet_nlp; max_time, callback)

# io = open("src/optim/results/lbfgs_MNIST_100.jl", "w")	
# write(io, string(accuracy))
# close(io)

# accuracy = Float32[]
# size_minibatch=20
# PSNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,1,40; pool_option=1), PK.Conv(5,5,40,30; pool_option=1), PK.SL(480,C,layer_PS[1]),  PK.SL(C*layer_PS[1],C,layer_PS[2]), PK.SL(C*layer_PS[2],C,layer_PS[3];f=identity))
# knet_nlp = KnetNLPModel(PSNet_PSLDP; data_train, data_test, size_minibatch)
# JSOSolvers.lbfgs(knet_nlp; max_time, callback)

# io = open("src/optim/results/lbfgs_MNIST_20.jl", "w")	
# write(io, string(accuracy))
# close(io)

#=
CIFAR10
=#
# (xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
# (xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

# C = 10 # number of classes
# layer_PS = [35,15,1] 


# data_train = (xtrn, ytrn)
# data_test = (xtst, ytst)

# accuracy = Float32[]
# size_minibatch=100
# PSNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# knet_nlp = KnetNLPModel(PSNet_PSLDP; data_train, data_test, size_minibatch)
# JSOSolvers.lbfgs(knet_nlp; max_time, callback)

# io = open("src/optim/results/lbfgs_CIFAR10_100.jl", "w")	
# write(io, string(accuracy))
# close(io)


# accuracy = Float32[]
# size_minibatch=20
# PSNet_PSLDP = PK.Chain_PSLDP(PK.Conv(5,5,3,60; pool_option=1), PK.Conv(5,5,60,30; pool_option=1), PK.SL(750, C, layer_PS[1]), PK.SL(C*layer_PS[1], C, layer_PS[2]), PK.SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# knet_nlp = KnetNLPModel(PSNet_PSLDP; data_train, data_test, size_minibatch)
# JSOSolvers.lbfgs(knet_nlp; max_time, callback)

# io = open("src/optim/results/lbfgs_CIFAR10_20.jl", "w")	
# write(io, string(accuracy))
# close(io)