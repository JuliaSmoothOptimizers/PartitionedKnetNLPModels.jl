using KnetNLPModels

abstract type PKnetChain end
abstract type PartitionedChain <: PKnetChain end 

# The structures <: PKnetChain are made to link the layers and express the loss function.
# PKnetChain assume a fiel layer.
no_dropout!(c::T,vec_dropout::Vector{Vector{Bool}}) where T <: PKnetChain =	map!(l-> l .= ones(Bool, length(l)), vec_dropout, c.layers)
no_dropout(c::T) where T <: PKnetChain = map(l -> ones(Bool,input(l)), c.layers) 

"""
    precompile_ps_struct(network<:Chain)

The function is called on a network defined by Dense/Sep_layer/Conv layers or any layer defining a `ps_struct(layer)`method.
It uses `ps_struct` successively onto layers to capture the variables each neuron depends on?.
It returns a precompiled PS structure as nested Vector of Integer, each of which informs about the variables (bias included) by each element function.
The dropout, is not tested for now.
"""
function precompile_ps_struct(c::T) where T <: PKnetChain 
	index = 0
	precompiled_ps_struct_layers = []
	for l in c.layers
		indexed_var_layer = ps_struct(l; index=index)
		index += indexed_var_layer.lengthlayer #on ajoute le nombre de variables de la couche pour obtenir une numérotation correcte des variables		
		push!(precompiled_ps_struct_layers, indexed_var_layer)
	end 
	precompiled_ps_struct = T(precompiled_ps_struct_layers...)
	return precompiled_ps_struct
end 

"""
    PS_deduction(c,dp)
    
Compute the partially separable structure of a network represented by the Chain c by performing a forward evaluation.
"""
function PS_deduction(c::T; dp=no_dropout(c)) where T <: PKnetChain
	inputs = input(c.layers[1])
	Dep = Vector{Vector{Int}}(map(i -> zeros(Int,0), 1:inputs)) # dépendance nulles de la taille des entrées
	length(dp)==length(c.layers) || error("size dropout does not match the network")
	for (index,l) in enumerate(c.layers)
		Dep = ps(l, Dep; dp=dp[index])		
	end
	Dep
end

PS(c::T) where T <: PKnetChain = PS_deduction(precompile_ps_struct(c))