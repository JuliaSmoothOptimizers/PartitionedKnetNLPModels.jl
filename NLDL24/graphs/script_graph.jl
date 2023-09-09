using Plots
using StatsBase

# when the default version is executed, then pwd() must be equal to ".../KnetNLPModels.jl/"
function one_graph(mean_std_methods = ["SGD", "Adam", "lbfgs"], single_run_methods = ["plse"];
                  initial_path="./NLDL24/graphs/raw_results/",
                  dataset= "MNIST",
                  run_number=7,
                  epoch_max=100,
                  debug=false)

  path = pwd() * initial_path * dataset * "/"
  debug && println("path: ", path)
  
  symbol_mean_std_methods = Symbol.(mean_std_methods)
  mean_symbol_table = (gradient_method -> Symbol("mean_" * gradient_method)).(mean_std_methods)
  std_symbol_table = (gradient_method -> Symbol("std_" * gradient_method)).(mean_std_methods)

  dic_means = Dict{Symbol, Vector{Float64}}()
  dic_stds = Dict{Symbol, Vector{Float64}}()
  for (index, symbol) in enumerate(symbol_mean_std_methods)
    include(path*String(symbol)*".jl")
    mean_symbol = mean_symbol_table[index]
    std_symbol = std_symbol_table[index]
    dic_means[symbol] = @eval $(mean_symbol)[1:min($epoch_max, length($mean_symbol))]
    dic_stds[symbol] = @eval $(std_symbol)[1:min($epoch_max, length($mean_symbol))]
  end

  

  p = plot(xlabel = "epochs", ylabel = "accuracy %")
  # pretty_symbol = (symbol -> String(symbol))
  for symbol in symbol_mean_std_methods
    println(symbol)
    plot!(p, dic_means[symbol], ribbon = dic_stds[symbol], fillalpha = 0.15, label = String(symbol))
  end
  
  # symbol_single_run_methods = Symbol.(single_run_methods) #Symbol("plse")

  size_dataset = (dataset == "MNIST" ? 60000 : 50000)
  range_step_plse = Int(size_dataset / (100)) # 100 = minbatchsize, the accuracy at every iterate

  
  accuracies_plse = -1 .* ones(Float32, epoch_max, run_number)
  for i in 1:run_number
    tmp = include(pwd() * initial_path * "plse/" * dataset * "/" * "plse_$(dataset)_100_run$(i).jl")
    accuracies_plse[1 : length(1:range_step_plse:min(epoch_max*range_step_plse+1, length(tmp))),i] = tmp[1:range_step_plse:min(epoch_max*range_step_plse+1, length(tmp))] 
    debug && initial_path * "plse/" * dataset * "/" * "plse_$(dataset)_100_run$(i).jl"
  end

  for i in 1:size(accuracies_plse,1)
    tmp = filter(val -> val != -1, accuracies_plse[i,:])    
    _mean = mean(tmp)
    for j in 1:size(accuracies_plse,2)
      (accuracies_plse[i,j] == -1) && (accuracies_plse[i,j] = _mean)
    end
  end

  (mean_plse, std_plse) = mean_and_std(accuracies_plse, 2)
  dic_mean = Dict{Symbol, Vector{Float32}}()
  dic_std = Dict{Symbol, Vector{Float32}}()
  dic_mean[:plse] = Vector(mean_plse[:,1])
  dic_std[:plse] = Vector(std_plse[:,1])
  plot!(p, dic_mean[:plse], ribbon = dic_std[:plse], fillalpha = 0.15, label = "plse")
  plot!(legend=:bottomright)

  savefigpath = path * "$(dataset).pdf"
  debug && println("savefig path: ", savefigpath)
  savefig(p, savefigpath) 
  
  return dic_means, dic_stds
end


function all_graphs(
  mean_std_methods::Vector{String}, 
  single_run_methods::Vector{String}, 
  dataset::Vector{String};
  initial_path="./NLDL24/graphs/raw_results/",
  debug=false)

  for string_dataset in dataset  
    one_graph(mean_std_methods, single_run_methods; initial_path, dataset=string_dataset, debug)
  end
end


mean_std_methods = ["SGD", "Adam", "lbfgs"]
# single_run_methods = ["lbfgs", "plse"]
single_run_methods = ["plse"]
dataset = ["MNIST", "CIFAR10"]
all_graphs(mean_std_methods, single_run_methods, dataset; debug=true)