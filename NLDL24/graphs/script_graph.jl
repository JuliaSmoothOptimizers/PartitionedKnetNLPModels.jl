using Plots
using StatsBase

# when the default version is executed, then pwd() must be equal to ".../KnetNLPModels.jl/"
function one_graph(mean_std_methods = ["SGD", "Adam", "lbfgs"];
                  initial_path="./NLDL24/graphs/raw_results/",
                  dataset= "MNIST",
                  run_number=1,
                  epoch_max=100,
                  debug=false)

  path = pwd() * initial_path * dataset * "/"
  debug && println("path: ", path)

  size_dataset = (dataset == "MNIST" ? 60000 : 50000)
  range_step = Int(size_dataset / (100)) # 100 = minbatchsize, the accuracy at every iterate
  
  accuracies_plsr1_AdaN = -1 .* ones(Float32, epoch_max, run_number)
  for i in 1:run_number
    tmp = include(pwd() * initial_path * dataset * "/plsr1_AdaN/plsr1_AdaN_$(dataset)_100_run$(i).jl")
    @show length(tmp[1:range_step:min(epoch_max*range_step+1, length(tmp))] )
    accuracies_plsr1_AdaN[1 : length(1:range_step:min(epoch_max*range_step+1, length(tmp))),i] = tmp[1:range_step:min(epoch_max*range_step+1, length(tmp))] 
    debug && initial_path * dataset * "/plsr1_AdaN/plsr1_AdaN_$(dataset)_100_run$(i).jl"
  end

  for i in 1:size(accuracies_plsr1_AdaN,1)
    tmp = filter(val -> val != -1, accuracies_plsr1_AdaN[i,:])    
    _mean = mean(tmp)
    for j in 1:size(accuracies_plsr1_AdaN,2)
      (accuracies_plsr1_AdaN[i,j] == -1) && (accuracies_plsr1_AdaN[i,j] = _mean)
    end
  end

  accuracies_plsr1 = -1 .* ones(Float32, epoch_max, run_number)
  for i in 1:run_number
    tmp = include(pwd() * initial_path * dataset * "/plsr1/plsr1_$(dataset)_100_run$(i).jl")
    accuracies_plsr1[1 : length(1:range_step:min(epoch_max*range_step+1, length(tmp))),i] = tmp[1:range_step:min(epoch_max*range_step+1, length(tmp))] 
    debug && initial_path * dataset * "/plsr1/plsr1_$(dataset)_100_run$(i).jl"
  end

  for i in 1:size(accuracies_plsr1,1)
    tmp = filter(val -> val != -1, accuracies_plsr1[i,:])    
    _mean = mean(tmp)
    for j in 1:size(accuracies_plsr1,2)
      (accuracies_plsr1[i,j] == -1) && (accuracies_plsr1[i,j] = _mean)
    end
  end

  (mean_plsr1_AdaN, std_plsr1_AdaN) = mean_and_std(accuracies_plsr1_AdaN, 2)  
  (mean_plsr1, std_plsr1) = mean_and_std(accuracies_plsr1, 2)  
  
  symbol_mean_std_methods = Symbol.(mean_std_methods)
  mean_symbol_table = (mean_ted_method -> Symbol("mean_" * mean_ted_method)).(mean_std_methods)
  std_symbol_table = (mean_ted_method -> Symbol("std_" * mean_ted_method)).(mean_std_methods)

  dic_means = Dict{Symbol, Vector{Float64}}()
  dic_stds = Dict{Symbol, Vector{Float64}}()
  
  dic_means[:plsr1_AdaN] = Vector(mean_plsr1_AdaN[:,1])
  dic_stds[:plsr1_AdaN] = Vector(std_plsr1_AdaN[:,1])
  dic_means[:plsr1] = Vector(mean_plsr1[:,1])
  dic_stds[:plsr1] = Vector(std_plsr1[:,1])
  
  for (index, symbol) in enumerate(symbol_mean_std_methods)
    include(path*String(symbol)*".jl")
    mean_symbol = mean_symbol_table[index]
    std_symbol = std_symbol_table[index]
    dic_means[symbol] = @eval $(mean_symbol)[1:min($epoch_max, length($mean_symbol))]
    dic_stds[symbol] = @eval $(std_symbol)[1:min($epoch_max, length($mean_symbol))]
  end

  p = plot(xlabel = "epochs", ylabel = "accuracy %")
  plot!(legend=:bottomright)
  pretty_symbol = (symbol -> replace(String(symbol), "plsr1_AdaN"=>"PLSR1_AdaN", "plsr1" => "PLSR1", "lbfgs"=>"LBFGS"))
  for symbol in vcat(symbol_mean_std_methods, Symbol[:plsr1_AdaN, :plsr1])
    println(symbol)
    if mapreduce(isnan, &, dic_stds[symbol])
      plot!(p, dic_means[symbol], label = pretty_symbol(symbol))
    else
      plot!(p, dic_means[symbol], ribbon = dic_stds[symbol], fillalpha = 0.15, label = pretty_symbol(symbol))
    end
  end
  
  savefigpath = path * "$(dataset).pdf"
  debug && println("savefig path: ", savefigpath)
  savefig(p, savefigpath) 
  
  return dic_means, dic_stds
end


function all_graphs(
  mean_std_methods::Vector{String}, 
  dataset::Vector{String};
  initial_path="./NLDL24/graphs/raw_results/",
  debug=false)

  for string_dataset in dataset  
    one_graph(mean_std_methods; initial_path, dataset=string_dataset, debug)
  end
end


mean_std_methods = ["SGD", "Adam", "lbfgs"]
# dataset = ["MNIST", "CIFAR10"]
# all_graphs(mean_std_methods, dataset; debug=true)

one_graph(mean_std_methods; dataset= "MNIST", run_number=2, epoch_max=50, debug=true)
one_graph(mean_std_methods; dataset= "CIFAR10", run_number=2, epoch_max=100, debug=true)