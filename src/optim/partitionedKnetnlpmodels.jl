using CUDA, Knet, KnetNLPModels, MLDatasets, NLPModels

mutable struct PartitionedKnetNLPModel{T <: Number, S, C <: PartitionedChain, Y <: Part_mat{T}, V} <: AbstractKnetNLPModel{T, S}
	meta :: NLPModelMeta{T, S}
	n :: Int
	C :: Int
	chain :: C
	counters :: Counters
	data_train
	data_test
	size_minibatch :: Int
  training_minibatch_iterator
  test_minibatch_iterator
  current_training_minibatch
  current_test_minibatch
	x0 :: S
	w :: S # == Vector{T}
	layers_g :: Vector{Param}
	nested_array :: V
	epv_g :: Elemental_pv{T}
	epv_s :: Elemental_pv{T}
	epv_work :: Elemental_pv{T}
	epv_res :: Elemental_pv{T}
	eplom_B :: Y
	table_indices :: Matrix{Vector{Int}} # choix arbitraire, à changer peut-être dans le futur
	name :: Symbol
	counter:: Counter_accuracy{T}
end

function PartitionedKnetNLPModel(chain_ANN :: P;
					size_minibatch :: Int=100,
					name=:plbfgs,
					data_train = begin (xtrn, ytrn) = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10; (xtrn, ytrn) end,
					data_test = begin (xtst, ytst) = MNIST.testdata(Float32); ytst[ytst.==0] .= 10; (xtst, ytst) end
					) where P <: PartitionedChain
	w0 = vector_params(chain_ANN)
	x0 = copy(w0)
	T = eltype(w0)
	n = length(w0)
	meta = NLPModelMeta(n, x0=w0)
	
	xtrn = data_train[1]
	ytrn = data_train[2]
	xtst = data_test[1]
	ytst = data_test[2]
	training_minibatch_iterator = create_minibatch(xtrn, ytrn, size_minibatch)	 	 	
	test_minibatch_iterator = create_minibatch(xtst, ytst, size_minibatch)
	current_training_minibatch = rand(training_minibatch_iterator)
	current_test_minibatch = rand(test_minibatch_iterator)

	C = chain_ANN.layers[end].out # assume a last layer separable 

	nested_array = build_nested_array_from_vec(chain_ANN, w0)
	layers_g = similar(params(chain_ANN)) # create a Vector of layer variables

	table_indices = build_listes_indices(chain_ANN)
	epv_g = partitioned_gradient(chain_ANN, current_training_minibatch, table_indices; n=n)
	epv_s = similar(epv_g)
	epv_work = similar(epv_g)
	epv_res = similar(epv_g)
	(name==:plbfgs) && (eplom_B = eplo_lbfgs_from_epv(epv_g))
	(name==:plsr1) && (eplom_B = eplo_lsr1_from_epv(epv_g))
	(name==:plse) && (eplom_B = eplo_lose_from_epv(epv_g))
	(name==:pbfgs) && (eplom_B = epm_from_epv(epv_g))
	(name==:psr1) && (eplom_B = epm_from_epv(epv_g))
	(name==:pse) && (eplom_B = epm_from_epv(epv_g))
	Y = typeof(eplom_B)
  V = typeof(nested_array)

	counter= Counter_accuracy(T)

	return PartitionedKnetNLPModel{T, Vector{T}, P, Y, V}(meta, n, C, chain_ANN, Counters(), data_train, data_test, size_minibatch, training_minibatch_iterator, test_minibatch_iterator, current_training_minibatch, current_test_minibatch, x0, w0, layers_g, nested_array, epv_g, epv_s, epv_work, epv_res, eplom_B, table_indices, name, counter)
end

# """
# 		set_size_minibatch!(knetnlp, size_minibatch)

# Change the size of the minibatchs of training and testing of the `knetnlp`.
# After a call of `set_size_minibatch!`, if one want to use a minibatch of size `size_minibatch` it must use beforehand `reset_minibatch_train!`.
# """
# function set_size_minibatch!(pknetnlp :: PartitionedKnetNLPModel, size_minibatch :: Int) 
# 	pknetnlp.size_minibatch = size_minibatch
# 	pknetnlp.minibatch_train = create_minibatch(pknetnlp.data_train[1], pknetnlp.data_train[2], pknetnlp.size_minibatch)
# 	pknetnlp.minibatch_test = create_minibatch(pknetnlp.data_test[1], pknetnlp.data_test[2], pknetnlp.size_minibatch)
# end

partitioned_gradient!(pknetnlp :: PartitionedKnetNLPModel; data=pknetnlp.current_training_minibatch) = partitioned_gradient!(pknetnlp.chain, data, pknetnlp.table_indices, pknetnlp.epv_g)

function mul_prod!(res:: Vector{T}, pknetnlp :: PartitionedKnetNLPModel{T,S,P}, v :: Vector{T}) where {T<:Number, S, P}
	eplom_B = pknetnlp.eplom_B
	epv_work = pknetnlp.epv_work
	PartitionedStructures.epv_from_v!(epv_work, v)
	epv_res = pknetnlp.epv_res
	PartitionedStructures.mul_epm_epv!(epv_res, eplom_B, epv_work)
	PartitionedStructures.build_v!(epv_res)
	res .= PartitionedStructures.get_v(epv_res)
end 
