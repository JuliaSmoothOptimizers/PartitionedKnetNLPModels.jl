module PartitionedKnetNLPModels
	using KnetNLPModels, PartitionedStructures

	include("ANN/_include.jl")
	include("optim/_include.jl")

	export Conv, Dense, Sep_layer, SL
	export Chain_NLL, Chain_PSLAP, Chain_PSLDP, Chain_PSLEP
	export accuracy

	export precompile_ps_struct, PS_deduction, PS

	export PartitionedChain
	export PartChainPSLDP
	export PartitionedKnetNLPModel

	export LBFGS, PLBFGS
	
end 