function one_sgd_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max), lr=0.0025)
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, " (seed $(index_seed)) initial accuracy: ", acc, " learning rate: ", lr)
  for i in 1:iter_max # 1 SGD epoch + accuracy check
    # xk = vector_params(chain)
    progress!(sgd(chain, ncycle(dtrn,1); lr))
    # xk1 = vector_params(chain)
    # @show norm(xk), norm(xk1), norm(xk1-xk)
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function one_nesterov_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 SGD epoch + accuracy check
    progress!(nesterov(chain, ncycle(dtrn,1)))
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function one_adadelta_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 SGD epoch + accuracy check
    progress!(adadelta(chain, ncycle(dtrn,1)))
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function one_momentum_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 SGD epoch + accuracy check
    progress!(momentum(chain, ncycle(dtrn,1)))
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function one_adagrad_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 SGD epoch + accuracy check
    xk = vector_params(chain)
    progress!(adagrad(chain, ncycle(dtrn,1)))
    xk1 = vector_params(chain)
    @show norm(xk), norm(xk1), norm(xk1-xk)    
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function one_rmsprop_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 SGD epoch + accuracy check
    progress!(rmsprop(chain, ncycle(dtrn,1)))
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end


function one_adam_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 Adam epoch + accuracy check    
    xk = vector_params(chain)
    progress!(adam(chain, ncycle(dtrn,1)))
    xk1 = vector_params(chain)
    @show norm(xk), norm(xk1), norm(xk1-xk)
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function seeded_sgd_trains(model_function, dtrn, dtst;
   iter_max=5,
   size_minibatch=100,
   name_architecture="undefined",
   name_dataset="undefined",
   max_seed=10,
   lr=0.1,
   )

  println("recap, iter_max: ", iter_max, " size_minibatch :", size_minibatch, " name_architecture: ", name_architecture, " name_dataset :", name_dataset)
  accuracy = Matrix{Float64}(undef, iter_max, max_seed)
  for seed in 1:max_seed
    view_acc = view(accuracy, 1:iter_max, seed)
    one_sgd_run!(model_function(), dtrn, dtst; iter_max, name=name_architecture, index_seed=seed, accuracy=view_acc, lr)
  end  
  println("accuracies % of $(name_architecture) on $(name_dataset) : \n", accuracy)

  (_mean, _std) = mean_and_std(accuracy, 2)
  mean = Vector(_mean[:,1])
  std = Vector(_std[:,1])

  println("mean % of $(name_architecture) on $(name_dataset) : \n", mean)
  println("std % of $(name_architecture) on $(name_dataset) : \n", std)

  io = open("NLDL24/results/SGD/minbatch$(size_minibatch)/$(name_dataset)/$(name_architecture).jl", "w")
  print(io, "accuracies_$(name_architecture)_$(name_dataset) = ", accuracy, "\n\n")
  print(io, "mean_$(name_architecture)_$(name_dataset) = ", mean, "\n\n")
  print(io, "std_$(name_architecture)_$(name_dataset) = ", std, "\n\n")

  close(io)
  return mean
end

function SGD_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, name_dataset = "MNIST")
  dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
  dtst = create_minibatch(xtst, ytst, size_minibatch)

  acc_SGD_LeNet_PSLDP = seeded_sgd_trains(LeNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_PSLDP", name_dataset, max_seed, lr=0.001)
  acc_SGD_PSNet_PSLDP = seeded_sgd_trains(PSNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP", name_dataset, max_seed, lr=0.001)
  acc_SGD_LeNet_NLL = seeded_sgd_trains(LeNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_NLL", name_dataset, max_seed, lr=0.1)  
  acc_SGD_PSNet_NLL = seeded_sgd_trains(PSNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_NLL", name_dataset, max_seed, lr=0.1)
  

  mean_accuracies = reshape(vcat(acc_SGD_LeNet_NLL, acc_SGD_LeNet_PSLDP, acc_SGD_PSNet_NLL, acc_SGD_PSNet_PSLDP), iter_max, 4)

  io = open("NLDL24/results/SGD/minbatch$(size_minibatch)/$(name_dataset)/mean_recap.jl", "w")
  print(io, "mean_accuracies_$(name_dataset) = ", mean_accuracies)
  close(io)
end


using Plots

# when the default version is executed, then pwd() must be equal to ".../KnetNLPModels.jl/"
function one_graph(size_minibatch, name_dataset; initial_path="/NLDL24/results/SGD", name_pair_loss_architecture=String[], debug=false)
  path = pwd() * initial_path * "/" * "minbatch$(size_minibatch)/$(name_dataset)/"
  debug && println("path: ", path)

  isempty(name_pair_loss_architecture) && error("No architecture-loss are compared")
  
  symbol_pair_loss_architecture = Symbol.(name_pair_loss_architecture)
  mean_symbol_table = (pair_architecture -> Symbol("mean_" * pair_architecture * "_$(name_dataset)")).(name_pair_loss_architecture)
  std_symbol_table = (pair_architecture -> Symbol("std_" * pair_architecture * "_$(name_dataset)")).(name_pair_loss_architecture)

  dic_means = Dict{Symbol, Vector{Float64}}()
  dic_stds = Dict{Symbol, Vector{Float64}}()
  for (index, symbol) in enumerate(symbol_pair_loss_architecture)
    include(path*String(symbol)*".jl")
    mean_symbol = mean_symbol_table[index]
    std_symbol = std_symbol_table[index]
    dic_means[symbol] = @eval $(mean_symbol)
    dic_stds[symbol] = @eval $(std_symbol)
  end

  p = plot(xlabel = "epochs", ylabel = "accuracy %")
  pretty_symbol = (symbol -> replace(String(symbol), "_" => "-", "PSLDP" => "PSL"))
  for symbol in symbol_pair_loss_architecture
    plot!(p, dic_means[symbol], ribbon = dic_stds[symbol] , fillalpha = 0.15, label = pretty_symbol(symbol))
  end
  plot!(legend=:bottomright)

  debug && println("savefig path: ", path * "minbatch$(size_minibatch)-$(name_dataset).pdf")
  savefig(p, path * "minbatch$(size_minibatch)-$(name_dataset).pdf") 
  
  return dic_means, dic_stds
end


function all_graphs(archicture_names::Vector{String}, 
                    loss_names::Vector{String},
                    minibatch_sizes::Vector{Int},
                    dataset_names::Vector{String};
                    debug=false)

  name_pair_loss_architecture = vec([name*"_"*loss for name in archicture_names, loss in loss_names])
  debug && println("pairs loss-architectures : ", name_pair_loss_architecture)
  
  for size_minibatch in minibatch_sizes
    for name_dataset in dataset_names
      println(name_dataset * " " * string(size_minibatch))
      dic_means, dic_stds = one_graph(size_minibatch, name_dataset; name_pair_loss_architecture, debug)
    end
  end
    
end
