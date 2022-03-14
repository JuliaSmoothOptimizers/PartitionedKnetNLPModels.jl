module PartitionedKnetNLPModels
	using KnetNLPModels, PartitionedStructures

	include("ANN/_include.jl")
	include("optim/_include.jl")

	export Conv, Dense, Sep_layer
	export Chain_NLL, Chain_PSLAP, Chain_PSLDP, Chain_PSLEP

end 