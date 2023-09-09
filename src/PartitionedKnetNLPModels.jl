module PartitionedKnetNLPModels
	using Knet
	using KnetNLPModels, PartitionedStructures

	include("ANN/_include.jl")
	include("optim/_include.jl")

	export Conv, Dense, Sep_layer, SL
	export Chain_NLL, Chain_PSLAP, Chain_PSLDP, Chain_PSLEP, Chain_PSLDP2

	export precompile_ps_struct, PS_deduction, PS

	export PartitionedChain
	export PartChainPSLDP, PartChainPSLDP2
	export PartitionedKnetNLPModel

	export LBFGS, PLBFGS, PLSR1
	export PUS, PLS, PLS_NA # include PBFGS, PSR1, PLSR1, PSE, PLSE
	
end 