using NLPModels
using KnetNLPModels


mutable struct PartitionedKnetNLPModel{T, S, C <: PartitionedChain} <: AbstractNLPModel{T, S}
	meta :: NLPModelMeta{T, S}
	n :: Int
	C :: Int
	chain :: C
	counters :: Counters
	data_train
	data_test
	size_minibatch :: Int
	minibatch_train
	minibatch_test
	current_minibatch_training
	current_minibatch_testing
	w :: S # == Vector{T}
	layers_g :: Vector{Param}
	nested_cuArray :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}
	epv_g :: Elemental_pv{T}
	epv_s :: Elemental_pv{T}
	epv_work :: Elemental_pv{T}
	eplom_B :: Elemental_plom_bfgs{T}
	table_indices :: Matrix{Vector{Int}} # choix arbitraire, à changer peut-être dans le futur
end


 function PartitionedKnetNLPModel(chain_ANN :: C;
            size_minibatch :: Int=100,
            data_train = begin (xtrn, ytrn) = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10; (xtrn, ytrn) end,
            data_test = begin (xtst, ytst) = MNIST.testdata(Float32); ytst[ytst.==0] .= 10; (xtst, ytst) end
            ) where C <: PartitionedChain
    w0 = vector_params(chain_ANN)
		T = eps(w0)
    n = length(w0)
    meta = NLPModelMeta(n, x0=w0)
    
    xtrn = data_train[1]
    ytrn = data_train[2]
    xtst = data_test[1]
    ytst = data_test[2]
    minibatch_train = create_minibatch(xtrn, ytrn, size_minibatch)	 	 	
    minibatch_test = create_minibatch(xtst, ytst, size_minibatch)
		current_minibatch_training = rand(minibatch_train)
		current_minibatch_testing = rand(minibatch_test)

    nested_array = build_nested_array_from_vec(chain_ANN, w0)
    layers_g = similar(params(chain_ANN)) # create a Vector of layer variables

		table_indices = build_listes_indices(chain_ANN)
		epv_g = partitioned_gradient(chain_ANN, current_minibatch_training, table_indices)
		epv_s = similar(epv_g)
		epv_work = similar(epv_g)
		eplom_B = eplom_lbfgs_from_epv(epv_grad)
		
    return PartitionedKnetNLPModel{T, Vector{T}, C}(meta, n, C, chain_ANN, Counters(), data_train, data_test, size_minibatch, minibatch_train, minibatch_test, current_minibatch_training, current_minibatch_testing, w0, layers_g, nested_array, pv_g, epv_s, epv_work, eplom_B, table_indices)
  end


	"""
			set_size_minibatch!(knetnlp, size_minibatch)
	
	Change the size of the minibatchs of training and testing of the `knetnlp`.
	After a call of `set_size_minibatch!`, if one want to use a minibatch of size `size_minibatch` it must use beforehand `reset_minibatch_train!`.
	"""
	function set_size_minibatch!(pknetnlp :: PartitionedKnetNLPModel, size_minibatch :: Int) 
		pknetnlp.size_minibatch = size_minibatch
		pknetnlp.minibatch_train = create_minibatch(pknetnlp.data_train[1], pknetnlp.data_train[2], pknetnlp.size_minibatch)
		pknetnlp.minibatch_test = create_minibatch(pknetnlp.data_test[1], pknetnlp.data_test[2], pknetnlp.size_minibatch)
	end

	partitioned_gradient!(pknetnlp :: PartitionedKnetNLPModel) = partitioned_gradient(pknetnlp.chain, pknetnlp.current_minibatch_training, pknetnlp.table_indices, pknetnlp.epv_g)

	"""
    	f = obj(nlp, x)

	Evaluate `f(x)`, the objective function of `nlp` at `x`.
	"""
	function NLPModels.obj(nlp :: KnetNLPModel{T, S, C}, w :: AbstractVector{T}) where {T, S, C}
		increment!(nlp, :neval_obj)
		set_vars!(nlp, w)
		res = nlp.chain(nlp.current_minibatch_training)
		f_w = sum(sum.(res))
		return f_w
	end

	"""
			g = grad!(nlp, x, g)

	Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
	"""
	function NLPModels.grad!(nlp :: KnetNLPModel{T, S, C}, w :: AbstractVector{T}, g :: AbstractVector{T}) where {T, S, C}
		@lencheck nlp.meta.nvar w g
		increment!(nlp, :neval_grad)
		set_vars!(nlp, w)  
		partitioned_gradient!(nlp)
		build_v!(nlp.epv_g)
		g .= get_v(nlp.epv_g)
		return g
	end


epv_tmp = epv_g
epv_g = partitioned_gradient()

add_epv!(epvg, minus_epv!(epv_tmp)) # compute epv_y