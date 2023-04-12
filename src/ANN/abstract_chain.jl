using KnetNLPModels

abstract type PartitionedChain <: KnetNLPModels.Chain end

# The structures <: KnetNLPModels.Chain are made to link the layers and express the loss function.
# KnetNLPModels.Chain assume a fiel layer.
no_dropout!(c::T, vec_dropout::Vector{Vector{Bool}}) where {T <: KnetNLPModels.Chain} =
  map!(l -> l .= ones(Bool, length(l)), vec_dropout, c.layers)
no_dropout(c::T) where {T <: KnetNLPModels.Chain} = map(l -> ones(Bool, input(l)), c.layers)

"""
    precompile_ps_struct(network<:Chain)
The function is called on a network defined by Dense/Sep_layer/Conv or an other layer that define also ps_struct(layer); index.
The result is a precompiled PS structure.
Il faut déterminer pour chaque couche une fonction ps_struct qui déterminer l'information nécessaire à prendre en compte pour déterminer la séparabilité partielle du réseau.
Sa forme est très proche de celle du réseau initial, mais chaque arc (ie variable) contient l'indice de la variable.
Attention il ne faut pas oublier de stocker les biais qui sont également des variables.
L'objectif est d'ensuite appelé une fonction évaluant rapidement la structure PS du réseau., par propagation des dépendances sur une évaluation backward.
Pour une analyse de la structure sous influence du dropout j'ai intégré des vecteur booléen indiquant neurones de la couche précédente étaient actifs ou non.
Cela se modélise par le vecteur dp dans PsDense
"""
function precompile_ps_struct(c::T) where {T <: KnetNLPModels.Chain}
  index = 0
  precompiled_ps_struct_layers = []
  for l in c.layers
    indexed_var_layer = ps_struct(l; index = index)
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
function PS_deduction(c::T; dp = no_dropout(c)) where {T <: KnetNLPModels.Chain}
  inputs = input(c.layers[1])
  Dep = Vector{Vector{Int}}(map(i -> zeros(Int, 0), 1:inputs)) # dépendance nulles de la taille des entrées
  length(dp) == length(c.layers) || error("size dropout does not match the network")
  for (index, l) in enumerate(c.layers)
    Dep = ps(l, Dep; dp = dp[index])
  end
  Dep
end

PS(c::T) where {T <: KnetNLPModels.Chain} = PS_deduction(precompile_ps_struct(c))
