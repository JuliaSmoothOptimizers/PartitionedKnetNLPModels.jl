using StatsBase 

function one_adam_run!(chain, dtrn, dtst;
                      iter_max=5,
                      name="undefined",
                      index_seed=0,
                      accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max),
                      lr=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      )
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 Adam epoch + accuracy check
    progress!(adam(chain, ncycle(dtrn,1); lr, beta1, beta2))
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function seeded_adam_trains(model_function, dtrn, dtst;
   iter_max=5,
   size_minibatch=100,
   name_architecture="undefined",
   name_dataset="undefined",
   max_seed=10,
   kwargs...)

  println("recap, iter_max: ", iter_max, " size_minibatch :", size_minibatch, " name_architecture: ", name_architecture, " name_dataset :", name_dataset)
  accuracy = Matrix{Float64}(undef, iter_max, max_seed)
  for seed in 1:max_seed
    view_acc = view(accuracy, 1:iter_max, seed)
    one_adam_run!(model_function(), dtrn, dtst; iter_max, name=name_architecture, index_seed=seed, accuracy=view_acc, kwargs...)
  end  
  println("accuracies % of $(name_architecture) on $(name_dataset) : \n", accuracy)

  (_mean, _std) = mean_and_std(accuracy, 2)
  mean = Vector(_mean[:,1])
  std = Vector(_std[:,1])

  println("mean % of $(name_architecture) on $(name_dataset) : \n", mean)
  println("std % of $(name_architecture) on $(name_dataset) : \n", std)

  io = open("NLDL24/results/Adam/minbatch$(size_minibatch)/$(name_dataset)/$(name_architecture).jl", "w")
  print(io, "accuracies_$(name_architecture)_$(name_dataset) = ", accuracy, "\n\n")
  print(io, "mean_$(name_architecture)_$(name_dataset) = ", mean, "\n\n")
  print(io, "std_$(name_architecture)_$(name_dataset) = ", std, "\n\n")

  close(io)
  return mean
end

function Adam_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, name_dataset = "MNIST")
  dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
  dtst = create_minibatch(xtst, ytst, size_minibatch)

  acc_Adam_LeNet_NLL = seeded_adam_trains(LeNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_NLL", name_dataset, max_seed)
  acc_Adam_LeNet_PSLDP = seeded_adam_trains(LeNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_PSLDP", name_dataset, max_seed)
  acc_Adam_PSNet_NLL = seeded_adam_trains(PSNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_NLL", name_dataset, max_seed)
  acc_Adam_PSNet_PSLDP = seeded_adam_trains(PSNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP", name_dataset, max_seed)

  mean_accuracies = reshape(vcat(acc_Adam_LeNet_NLL, acc_Adam_LeNet_PSLDP, acc_Adam_PSNet_NLL, acc_Adam_PSNet_PSLDP), iter_max, 4)

  io = open("NLDL24/results/Adam/minbatch$(size_minibatch)/$(name_dataset)/mean_recap.jl", "w")
  print(io, "mean_accuracies_$(name_dataset) = ", mean_accuracies)
  close(io)
end


function Adam_test_hyper_parameters(xtrn, ytrn,
                                    xtst, ytst,
                                    iter_max, size_minibatch,
                                    max_seed;
                                    name_dataset = "MNIST",
                                    lrs = [0.001],
                                    beta1s = [0.9],
                                    beta2s = [0.999],
                                    )
  dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
  dtst = create_minibatch(xtst, ytst, size_minibatch)

  string_decimal_selector(n::Number) = replace(string(n), "0."=> "") 

  for lr in lrs
    for beta1 in beta1s
      for beta2 in beta2s
        seeded_adam_trains(PSNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP_" * string_decimal_selector(lr) * "_" * string_decimal_selector(beta1) * "_" * string_decimal_selector(beta2), name_dataset, max_seed, lr, beta1, beta2)
      end
    end
  end

end

using Plots

# when the default version is executed, then pwd() must be equal to ".../KnetNLPModels.jl/"
function one_graph(size_minibatch, name_dataset; initial_path="/NLDL24/results/Adam", name_pair_loss_architecture=String[], debug=false)
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

function all_hyperparameters(;
  initial_path="/NLDL24/results/Adam_test_hyperparameters/",
  lrs, beta1s, beta2s, debug=false)

  path = pwd() * initial_path
  debug && println("path: ", path)

  string_decimal_selector(n::Number) = replace(string(n), "0."=> "") 

  name_run = [ "PSNet_PSLDP_"*string_decimal_selector(lr)*"_"*string_decimal_selector(beta1)*"_"*string_decimal_selector(beta2) for lr in lrs, beta1 in beta1s, beta2 in beta2s]

  symbol_pair_loss_architecture = Symbol.(name_run)
  mean_symbol_table = (run -> Symbol("mean_" * run * "_CIFAR10")).(name_run)
  std_symbol_table = (run -> Symbol("std_" * run * "_CIFAR10")).(name_run)

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
  pretty_symbol = (symbol -> replace(String(symbol), "_" => "-", "PSNet_PSLDP" => ""))
  for symbol in symbol_pair_loss_architecture
    plot!(p, dic_means[symbol], ribbon = dic_stds[symbol] , fillalpha = 0.15, label = pretty_symbol(symbol))
  end
  plot!(legend=:outerbottomright)

  debug && println("savefig path: ", path * "test.pdf")
  savefig(p, path * "test.pdf") 

  return dic_means, dic_stds
end