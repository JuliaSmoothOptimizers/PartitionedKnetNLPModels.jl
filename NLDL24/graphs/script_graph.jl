using Plots

# when the default version is executed, then pwd() must be equal to ".../KnetNLPModels.jl/"
function one_graph(gradient_methods = ["SGD", "Adam"], QN_methods = ["lbfgs", "plse"];
                  initial_path="./NLDL24/graphs/raw_results/",
                  dataset= "MNIST",
                  epoch_max=50,
                  debug=false)

  path = pwd() * initial_path * dataset * "/"
  debug && println("path: ", path)
  
  symbol_gradient_methods = Symbol.(gradient_methods)
  mean_symbol_table = (gradient_method -> Symbol("mean_" * gradient_method)).(gradient_methods)
  std_symbol_table = (gradient_method -> Symbol("std_" * gradient_method)).(gradient_methods)

  dic_means = Dict{Symbol, Vector{Float64}}()
  dic_stds = Dict{Symbol, Vector{Float64}}()
  for (index, symbol) in enumerate(symbol_gradient_methods)
    include(path*String(symbol)*".jl")
    mean_symbol = mean_symbol_table[index]
    std_symbol = std_symbol_table[index]
    dic_means[symbol] = @eval $(mean_symbol)
    dic_stds[symbol] = @eval $(std_symbol)
  end

  p = plot(xlabel = "epochs", ylabel = "accuracy %")
  # pretty_symbol = (symbol -> String(symbol))
  for symbol in symbol_gradient_methods
    println(symbol)
    plot!(p, dic_means[symbol], ribbon = dic_stds[symbol], fillalpha = 0.15, label = String(symbol))
  end
  
  symbol_QN_methods = Symbol.(QN_methods)
  accuracies_symbol_table = (QN_method -> Symbol("accuracies_" * QN_method)).(QN_methods)

  dic_accuracies = Dict{Symbol, Vector{Float32}}()
  for (index, symbol) in enumerate(symbol_QN_methods)
    include(path*String(symbol)*".jl")
    accuracy_symbol = accuracies_symbol_table[index]
    dic_accuracies[symbol] = @eval $(accuracy_symbol)    
  end

  size_dataset = (dataset == "MNIST" ? 60000 : 50000)
  
  # LBFGS preprocess
  if "lbfgs" in QN_methods
    range_step_lfbgs = Int(size_dataset / (100 * 10)) # 100 = minbatchsize, the accuracy is retrived every 10 iterates
    dic_accuracies[Symbol("lbfgs")] = dic_accuracies[Symbol("lbfgs")][1:range_step_lfbgs:min(epoch_max*range_step_lfbgs+1, length(dic_accuracies[Symbol("lbfgs")]))] 
  end
  # PLSEs preprocess
  if "plse" in QN_methods
    range_step_plse = Int(size_dataset / (100)) # 100 = minbatchsize, the accuracy at every iterate
    dic_accuracies[Symbol("plse")] = dic_accuracies[Symbol("plse")][1:range_step_plse:min(epoch_max*range_step_plse+1, length(dic_accuracies[Symbol("plse")]))] 
  end

  for symbol in symbol_QN_methods
    plot!(p, dic_accuracies[symbol], label = String(symbol))
  end
  plot!(legend=:bottomright)

  savefigpath = path * "$(dataset).pdf"
  debug && println("savefig path: ", savefigpath)
  savefig(p, savefigpath) 
  
  return dic_means, dic_stds
end


function all_graphs(
  gradient_methods::Vector{String}, 
  QN_methods::Vector{String}, 
  dataset::Vector{String};
  initial_path="./NLDL24/graphs/raw_results/",
  debug=false)

  for string_dataset in dataset  
    one_graph(gradient_methods, QN_methods; initial_path, dataset=string_dataset, debug)
  end
end


gradient_methods = ["SGD", "Adam"]
QN_methods = ["lbfgs", "plse"]
dataset = ["MNIST", "CIFAR10"]
all_graphs(gradient_methods, QN_methods, dataset; debug=true)