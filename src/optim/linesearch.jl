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

	println("PQN update using truncated conjugate-gradient, α=", α)
	(x,iter) = LSCG(nlp, B; α, x, ∇f₀, max_eval=max_eval, max_time=max_time, kwargs...)

	printing && (io = open("src/optim/results/linesearch_" * string(nlp.name) * ".jl", "w+")	)
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
	is_KnetNLPModel::Bool=false,
	verbose::Bool=true,
	data::Bool=true,
  subsolver=CgSolver(B, x),
	kwargs...,
	) where T <: Number

	iter = 0 # ≈ k
	gₖ = Vector{T}(undef,n); gₖ = ∇f₀
	gₜₘₚ = similar(gₖ)
	sₖ = similar(gₖ)
  xtmp = similar(gₖ)
	∇fNorm2 = nrm2(n, ∇f₀)

	fₖ = NLPModels.obj(nlp, x; α)
	verbose && @printf "iter temps fₖ      ||gₖ||    ||sₖ||    β     /100 \n" 

	# cgtol = one(T)  # Must be ≤ 1.
	# cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
  atol = (T)(0)
  cgtol = (T)(0)

	ρₖ = -1
  β = -1.

	# stop condition
	absolute(n,gₖ,ϵ) = nrm2(n,gₖ) > ϵ
	relative(n,gₖ,ϵ,∇fNorm2) = nrm2(n,gₖ) > ϵ * ∇fNorm2
	_max_iter(iter, max_iter) = iter < max_iter
	_max_time(start_time) = (time() - start_time) < max_time
	while absolute(n,gₖ,ϵ) && relative(n,gₖ,ϵ,∇fNorm2) && _max_iter(iter, max_iter) & _max_time(start_time) && isnan(ρₖ)==false# stop condition
		iter += 1    
    fₖ = NLPModels.obj(nlp, x; α)
    NLPModels.grad!(nlp, x, gₖ; α)

		(verbose || data ) && (acc = KnetNLPModels.accuracy(nlp))
    verbose && mod(iter, 10) == 0 && @printf "iter temps fₖ      ||gₖ||    ||sₖ||    β     /100 \n" 
		verbose && @printf "%3d %4g %8.1e %7.1e %7.1e %7.1e %8.3e " iter (time() - start_time) fₖ norm(gₖ, 2) norm(sₖ, 2) β acc 
		data && push_acc!(nlp.counter, acc)
   
    gₜₘₚ .= .- gₖ
		# Krylov.cg!(subsolver, B, gₜₘₚ, atol=T(atol), rtol=cgtol, radius = T(Δ))
    Krylov.cg!(subsolver, B, gₜₘₚ, atol=T(atol), rtol=cgtol, linesearch=true)
		sₖ .= solution(subsolver)  # result of the linear system solved by Krylov.cg    

		β, fₖ₊₁ = linesearch!(x, fₖ, sₖ, nlp, gₖ; vec=xtmp, α) # we compute the ratio
		# step acceptance + update f,g
    if β != -1. # the step exists
			epv_from_epv!(nlp.epv_work, nlp.epv_g)
			NLPModels.grad!(nlp, x, gₖ; α)
			minus_epv!(nlp.epv_work)
			add_epv!(nlp.epv_g, nlp.epv_work) # compute epv_y
			epv_from_v!(nlp.epv_s, sₖ)
			# PartitionedStructures.update!(nlp.eplom_B, nlp.epv_work, nlp.epv_s; name=nlp.name, verbose=false)
      # PartitionedStructures.update!(nlp.eplom_B, nlp.epv_work, nlp.epv_s; name=nlp.name, verbose=true, reset=4)
      PartitionedStructures.update!(nlp.eplom_B, nlp.epv_work, nlp.epv_s; name=nlp.name, verbose=false, reset=4)
		  fₖ = fₖ₊₁
			@printf "✅\n"
		else
			@printf "❌\n"
		end				

		# change the minibatch
		is_KnetNLPModel && reset_minibatch_train!(nlp)
		
	end
  acc = KnetNLPModels.accuracy(nlp)
  amount_training_minibatches = length(nlp.training_minibatch_iterator)
  @printf "After %3d iterations, which is approximating %4g epochs\n" iter iter/amount_training_minibatches  
  @printf "iter temps fₖ   /100 \n" 
  @printf "%3d %4g %8.1e %8.3e " iter (time() - start_time) fₖ acc 
	
	return (x, iter)
end

well_defined(β) = β > sqrt(eps(β))
armijo(fₖ, fₖ₊₁, β, t) = fₖ - fₖ₊₁ ≥ β *  t

"""
    linesearch!x::Vector{T}, fₖ::T, sₖ::Vector{T}, nlp::AbstractNLPModel{T}, gₖ::AbstractVector{T}; vec=similar(x), c=0.05, τ=2/3, init=1., β = init, α=0.05 ) where T <: Number

Find the step length β.
"""
function linesearch!(x::Vector{T},
  fₖ::T,
  sₖ::Vector{T},
  nlp::AbstractNLPModel,
  gₖ::AbstractVector{T};
  vec=similar(x),
  c=0.05,
  τ=2/3,
  init=1.,
  β = init,
  α=0.05
) where T <: Number
  build_v!(nlp.epv_g)
  m = dot(gₖ, sₖ)
  t = - c * m 
    
  vec .= x .+ β .* sₖ # xₖ₊₁
  fₜₘₚ = fₖ
  fₖ₊₁ = NLPModels.obj(nlp, vec; α)
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁

  # determining if β may be > 1
  while armijo(fₖ, fₖ₊₁, β, t) && fₖ₊₁ < fₜₘₚ # stops when armijo fails
    β = 1/τ * β    
    vec .= x .+ β .* sₖ # xₖ₊₁
    fₜₘₚ = fₖ₊₁
    fₖ₊₁ = NLPModels.obj(nlp, vec; α)
    # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  end

  if β > init # may be changed    
    β = τ * β
    vec .= x .+ β .* sₖ # xₖ₊₁
    fₖ₊₁ = NLPModels.obj(nlp, vec; α)    
  else
    while well_defined(β)
      vec .= x .+ β .* sₖ # xₖ₊₁
      fₖ₊₁ = NLPModels.obj(nlp, vec; α)
      armijo(fₖ, fₖ₊₁, β, t) ? break : β = τ * β 
      # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
    end  
  end

  if well_defined(β) # update x, fₖ₊₁ is up to date
    x .= vec    
  else # set fₖ₊₁ to fₖ
    β = -1.
    @printf "❌ β too small! "
    fₖ₊₁ = fₖ
  end
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  # @printf "β = %8.3e ≥ 1, fₖ = %8.3e, fₖ₊₁ = %8.3e \n" β fₖ fₖ₊₁
	return β, fₖ₊₁
end