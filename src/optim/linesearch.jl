using LinearOperators, NLPModels, LinearAlgebra, LinearAlgebra.BLAS, Krylov
using Printf, SolverTools, SolverCore


PLS(nlp :: PartitionedKnetNLPModel; kwargs...) = partitioned_linesearch(nlp; is_KnetNLPModel=true, kwargs...)
function partitioned_linesearch(nlp :: AbstractNLPModel;
	x::AbstractVector=copy(nlp.meta.x0),
	T::DataType = eltype(x),
	kwargs...)
	n = nlp.meta.nvar
	B(nlp) = LinearOperator(T, n, n, true, true, (res, v)-> mul_prod!(res, nlp, v))
	type_update = nlp.name
	println("LinearOperator{$T} ✅ from $type_update")
	return partitioned_linesearch(nlp, B(nlp); x=x, kwargs...)
end

function partitioned_linesearch(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
  x::AbstractVector=copy(nlp.meta.x0),
	max_eval :: Int=10000,
	max_iter::Int=10000,
	start_time::Float64=time(),
	max_time :: Float64=30.0,
	ϵ::Float64= 1e-6,
  printing::Bool=false,
  α=0.05,
	kwargs...) where T <: Number

	x₀ = x
	n = length(x₀)
  ∇f₀ = similar(x₀)
	NLPModels.grad!(nlp, x₀, ∇f₀; α)
	∇fNorm2 = norm(∇f₀,2)

	println(string(nlp.name) * "update using truncated conjugate-gradient, α=", α, " ϵ=", ϵ)
	(x,iter) = LSCG(nlp, B; α, x, ∇f₀, max_eval=max_eval, max_time=max_time, max_iter, ϵ, kwargs...)

	printing && (io = open("src/optim/results/linesearch_" * string(nlp.name) * ".jl", "w")	)
	printing && (write(io, string(nlp.counter.acc)))
	printing && (close(io))
	
	Δt = time() - start_time
  g = similar(x)
	NLPModels.grad!(nlp, x₀, g; α)
	nrm_grad = norm(g,2)

	absolute(n,gₖ,ϵ) = nrm2(n,gₖ) < ϵ
	relative(n,gₖ,ϵ,∇fNorm2) = nrm2(n,gₖ) < ϵ * ∇fNorm2
	_max_iter(iter, max_iter) = iter >= max_iter
	_max_time(start_time) = (time() - start_time) >= max_time

	if absolute(n,g,ϵ) || relative(n,g,ϵ,∇fNorm2)
		status = :first_order
		println("point stationnaire ✅")
	elseif _max_iter(iter, max_iter)
		status = :max_eval
		println("Max eval ❌")
	elseif _max_time(start_time)
		status = :max_time
		println("Max time ❌")
	else
		status = :unknown
		println("Unknown ❌")
	end
	return GenericExecutionStats(nlp;
                         status, 
												 solution=x,
												 iter=iter,
												 dual_feas = nrm_grad,
												 objective = NLPModels.obj(nlp, x),
												 elapsed_time = Δt,
												)
end


"""
    LSCG(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
	x::AbstractVector{T}=copy(nlp.meta.x0),
	n::Int=nlp.meta.nvar,
	max_eval::Int=10000,
	max_iter::Int=10000,
	max_time :: Float64=30.0,
	atol::Real=√eps(eltype(x)),
	rtol::Real=√eps(eltype(x)),
	start_time::Float64=time(),
	η::Float64=1e-3,
	η₁::Float64=0.75, # > η	
	ϵ::Float64=1e-6,
	δ::Float64=1.,
	ϕ::Float64=2.,
  α::Float64=0.05,
	∇f₀::AbstractVector=NLPModels.grad(nlp, nlp.meta.x0; ∇f₀),
	iter_print::Int64=Int(floor(max_iter/100)),
	is_KnetNLPModel::Bool=false,
	verbose::Bool=true,
	data::Bool=true,
  subsolver=CgSolver(B, x),
	kwargs...,
	) where T <: Number

Apply a inexact linesearch using CG.
Event if `nlp` is typed as an `AbstractNLPModel` it must be a `PartitionedKnetNLPModel`.
"""
function LSCG(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
	x::AbstractVector{T}=copy(nlp.meta.x0),
	n::Int=nlp.meta.nvar,
	max_eval::Int=10000,
	max_iter::Int=10000,
	max_time :: Float64=30.0,
	atol::Real=√eps(eltype(x)),
	rtol::Real=√eps(eltype(x)),
	start_time::Float64=time(),
	η::Float64=1e-3,
	η₁::Float64=0.75, # > η	
	ϵ::Float64=1e-6,
	δ::Float64=1.,
	ϕ::Float64=2.,
  α::Float64=0.05,
	∇f₀::AbstractVector=NLPModels.grad(nlp, nlp.meta.x0; ∇f₀),
	iter_print::Int64=Int(floor(max_iter/100)),
	is_KnetNLPModel::Bool=true,
	verbose::Bool=true,
	data::Bool=true,
  linesearch=true,
  linesearch_option=:basic,
  subsolver=CgSolver(B, x),
	kwargs...,
	) where T <: Number

	iter = 0 # ≈ k
	gₖ = Vector{T}(undef,n); gₖ = ∇f₀
	gₜₘₚ = similar(gₖ)
	sₖ = similar(gₖ)
  xtmp = similar(gₖ)
	∇fNorm2 = nrm2(n, ∇f₀)
  scaled_step = similar(gₖ)

	fₖ = NLPModels.obj(nlp, x; α)
	verbose && @printf "iter temps fₖ      ||gₖ||    ||sₖ||    β     /100 \n" 

	# cgtol = one(T)  # Must be ≤ 1.
	# cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
  atol = (T)(0)
  cgtol = (T)(0)
	
  β = -1.

	# stop condition
	absolute(n,gₖ,ϵ) = nrm2(n,gₖ) > ϵ
	relative(n,gₖ,ϵ,∇fNorm2) = nrm2(n,gₖ) > ϵ * ∇fNorm2
	_max_iter(iter, max_iter) = iter < max_iter
	_max_time(start_time) = (time() - start_time) < max_time
	while absolute(n,gₖ,ϵ) && relative(n,gₖ,ϵ,∇fNorm2) && _max_iter(iter, max_iter) & _max_time(start_time) # stop condition
		iter += 1    
    fₖ = NLPModels.obj(nlp, x; α)
    NLPModels.grad!(nlp, x, gₖ; α)

		(verbose || data ) && (acc = KnetNLPModels.accuracy(nlp))
    verbose && mod(iter, 10) == 0 && @printf "iter temps fₖ      ||gₖ||    ||sₖ||    β     /100 -- PLS_%s \n" string(nlp.name)
		verbose && @printf "%3d %4g %8.1e %7.1e %7.1e %7.1e %8.3e " iter (time() - start_time) fₖ norm(gₖ, 2) norm(sₖ, 2) β acc 
		data && push_acc!(nlp.counter, acc)
   
    gₜₘₚ .= .- gₖ
		# Krylov.cg!(subsolver, B, gₜₘₚ, atol=T(atol), rtol=cgtol, radius = T(Δ))
    Krylov.cg!(subsolver, B, gₜₘₚ, atol=T(atol), rtol=cgtol, linesearch=true)
		sₖ .= solution(subsolver)  # result of the linear system solved by Krylov.cg    

    # x is returned as xₖ₊₁
		if linesearch 
      (linesearch_option == :basic) && (β = basic_linesearch(x, fₖ, sₖ, nlp, gₜₘₚ; vec=xtmp, α)) # we compute the ratio
      (linesearch_option == :backtracking) && (β = backtracking_linesearch(x, fₖ, sₖ, nlp, gₜₘₚ; vec=xtmp, α)) # we compute the ratio
    else
      β = 1.
    end


		# step acceptance + update f,g
    if β != -1. # the step exists
      x .= x .+ β .* sₖ

			epv_from_epv!(nlp.epv_work, nlp.epv_g) # epv_work -> current gradient (gₖ)
			minus_epv!(nlp.epv_work) # - gₖ

      NLPModels.grad!(nlp, x, gₖ; α) # update nlp.epv_g and gₖ to gₖ₊₁
			add_epv!(nlp.epv_g, nlp.epv_work) # compute epv_y, gₖ₊₁ - gₖ
      scaled_step .= β .* sₖ
			epv_from_v!(nlp.epv_s, scaled_step)
			
      PartitionedStructures.update!(nlp.eplom_B, nlp.epv_work, nlp.epv_s; name=nlp.name, verbose=false, reset=4)		  
			@printf "✅\n"
		else
			@printf "❌\n"
		end				

		# change the minibatch
		is_KnetNLPModel && minibatch_next_train!(nlp)
		
	end
  acc = KnetNLPModels.accuracy(nlp)
  amount_training_minibatches = length(nlp.training_minibatch_iterator)
  @printf "After %3d iterations, which is approximating %4g epochs\n" iter iter/amount_training_minibatches  
  @printf "iter temps fₖ   /100 \n" 
  @printf "%3d %4g %8.1e %8.3e " iter (time() - start_time) fₖ acc 
	
	return (x, iter)
end
