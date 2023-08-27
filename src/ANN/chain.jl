# Define some partially-separable loss functions


#  Chain_NLL, utilisée pour faire le lien entre les différents layers.
# Son evaluation contient également la fonction de perte negative log likehood
# Elle est également utilisé afin de précompilé la structure PS d'un réseau
struct Chain_NLL <: PKnetChain
	layers
	Chain_NLL(layers...) = new(layers)
end
(c::Chain_NLL)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_NLL)(x,y) = nll(c(x),y) # nécessaire
(c::Chain_NLL)(data :: Tuple{T1,T2}) where {T1,T2} = nll(c(data[1]), data[2], average=true)
(c::Chain_NLL)(d::Knet.Data) = nll(c; data=d, average=true) 
# no_dropout(c::Chain_NLL)=map(l -> ones(Bool,input(l)), c.layers) 
# à utiliser une fois que vec_dropout a été correctement initialisé
# no_dropout!(c::Chain_NLL,vec_dropout::Vector{Vector{Bool}}) =	map!(l-> l .= ones(Bool, length(l)), vec_dropout, c.layers)


mutable struct Chain_PSLAP <: PKnetChain
	layers
	Chain_PSLAP(layers...) = new(layers)
end
(c::Chain_PSLAP)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLAP)(x,y) = PSLAP(c(x),y)
(c::Chain_PSLAP)(d::Knet.Data) = PSLAP(c; data=d, average=true)
function PSLAP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLAP(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function PSLAP(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = exp.(scores .- maximum(scores, dims=1)) # enlever cette ligne fonctionne moins bien fonction	
	acc = sum(.- log.(scores[indices]))
	ninstances = length(labels); y1 = size(scores,1) 
	tmp = [1:(ninstances*y1);]; splice!(tmp, indices)
	acc += sum(scores[tmp])
	average ? (acc / length(labels)) : (acc, length(labels))
end

#PSLEP
mutable struct Chain_PSLEP <: PKnetChain
	layers
	Chain_PSLEP(layers...) = new(layers)
end
(c::Chain_PSLEP)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLEP)(x,y) = PSLEP(c(x),y)
(c::Chain_PSLEP)(d::Knet.Data) = PSLEP(c; data=d, average=true)
function PSLEP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLEP(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function PSLEP(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = exp.(scores .- maximum(scores, dims=1)) # enlever cette ligne fonctionne moins bien fonction	
	acc = sum(.- log.(scores[indices]))
	average ? (acc / length(labels)) : (acc, length(labels))
end

#PSLDP
mutable struct Chain_PSLDP <: PKnetChain
	layers
	Chain_PSLDP(layers...) = new(layers)
end
(c::Chain_PSLDP)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLDP)(x,y) = PSLDP(c(x),y)
(c::Chain_PSLDP)(data :: Tuple{T1,T2}) where {T1,T2} = _PSLDP(c; data=data, average=true)
(c::Chain_PSLDP)(d::Knet.Data) = PSLDP(c; data=d, average=true)
function PSLDP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLDP(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function _PSLDP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	(x,y) = data
	(z,n) = PSLDP(model(x; o...), y; dims=dims, average=false) 
	sum += z; cnt += n
	average ? sum / cnt : (sum, cnt)
end
function PSLDP(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	# scores = exp.(scores .- reshape(scores[indices], 1, length(indices))) # diminue par les scores par celui que l'on cherche à obtenir  
  # scores = (x -> x^2).(exp.(scores .- reshape(scores[indices],1, length(indices)))) .- 1. # test
  scores = (x -> x^2).(exp.(scores .- reshape(scores[indices],1, length(indices))))
  # scores = (exp.(scores .- reshape(scores[indices],1, length(indices))))
	# absence de garantie < 1
	acc = sum(scores)
	average ? (acc / length(labels)) : (acc, length(labels))
end

#PSLDP2
mutable struct Chain_PSLDP2 <: PKnetChain
	layers
	Chain_PSLDP2(layers...) = new(layers)
end
(c::Chain_PSLDP2)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLDP2)(x,y) = PSLDP2(c(x),y)
(c::Chain_PSLDP2)(data :: Tuple{T1,T2}) where {T1,T2} = _PSLDP2(c; data=data, average=true)
(c::Chain_PSLDP2)(d::Knet.Data) = PSLDP2(c; data=d, average=true)
function PSLDP2(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLDP2(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function _PSLDP2(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	(x,y) = data
	(z,n) = PSLDP2(model(x; o...), y; dims=dims, average=false) 
	sum += z; cnt += n
	average ? sum / cnt : (sum, cnt)
end
function PSLDP2(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	# scores = exp.(scores .- reshape(scores[indices], 1, length(indices))) # diminue par les scores par celui que l'on cherche à obtenir  
  # scores = (x -> x^2).(exp.(scores .- reshape(scores[indices],1, length(indices)))) .- 1. # test
  
  # losses = (x -> x^2).(exp.(scores .- reshape(scores[indices], 1, length(indices))))
  
  size_NN_output = size(scores, 1) # 10 for MNIST-CIFAR10, 100 for CIFAR100
  indice_max = mapreduce(i->i, *, size(scores))
  losses = sum(index -> (exp(scores[index] - scalar_factor(index, size_NN_output, indices) * scores[indices[index_indices(index, size_NN_output)]]))^2, 1:indice_max)
  counter = sum(index -> scalar_factor(index, size_NN_output, indices), 1:indice_max)
  # @show counter

	# absence de garantie < 1
	average ? (losses / length(labels)) : (losses, length(labels))
end

index_indices(index, size_NN_output) = Int(((index - 1 - (index-1) % size_NN_output)) / size_NN_output +1 )
scalar_factor(index, size_NN_output, indices) = indices[index_indices(index, size_NN_output)] == index ? 2 : 1