using LinearOperators, NLPModels, LinearAlgebra, LinearAlgebra.BLAS, Krylov
using Printf, SolverTools, SolverCore


PUS(nlp :: PartitionedKnetNLPModel; kwargs...) = partitioned_update_solver(nlp; is_KnetNLPModel=true, kwargs...)
function partitioned_update_solver(nlp :: AbstractNLPModel;
	x::AbstractVector=copy(nlp.meta.x0),
	T::DataType = eltype(x),
	kwargs...)
	n = nlp.meta.nvar
	B(nlp) = LinearOperator(T, n, n, true, true, (res, v)-> mul_prod!(res, nlp, v))
	type_update = nlp.name
	println("LinearOperator{$T} ✅from $type_update")
	return partitioned_update_solver(nlp, B(nlp); x=x, kwargs...)
end

function partitioned_update_solver(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
	max_eval :: Int=10000,
	max_iter::Int=10000,
	start_time::Float64=time(),
	max_time :: Float64=30.0,
	ϵ::Float64= 1e-6,
	kwargs...) where T <: Number

	x₀ = nlp.meta.x0
	n = length(x₀)
	∇f₀ = NLPModels.grad(nlp, x₀)
	∇fNorm2 = norm(∇f₀,2)

	println("Start trust-region PQN update using truncated conjugate-gradient")
	(x,iter) = TRCG_KNLP_PUS(nlp, B; max_eval=max_eval, max_time=max_time, kwargs...)

	io = open("src/optim/results/accuracy_PUS_" * string(nlp.name) * ".txt", "w+")	
	write(io, string(nlp.counter.acc))
	close(io)
	
	Δt = time() - start_time
	g = NLPModels.grad(nlp, x)
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
	return GenericExecutionStats(status, nlp,
												 solution=x,
												 iter=iter,
												 dual_feas = nrm_grad,
												 objective = NLPModels.obj(nlp, x),
												 elapsed_time = Δt,
												)
end


function TRCG_KNLP_PUS(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
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
	Δ::Float64=1.,
	ϵ::Float64=1e-6,
	δ::Float64=1.,
	ϕ::Float64=2.,
	∇f₀::AbstractVector=NLPModels.grad(nlp, nlp.meta.x0),
	iter_print::Int64=Int(floor(max_iter/100)),
	is_KnetNLPModel::Bool=false,
	verbose::Bool=true,
	data::Bool=true,
	kwargs...,
	) where T <: Number

	iter = 0 # ≈ k
	gₖ = Vector{T}(undef,n); gₖ = ∇f₀
	gₜₘₚ = similar(gₖ)
	sₖ = similar(gₖ)
  xtmp = similar(gₖ)
	∇fNorm2 = nrm2(n, ∇f₀)

	fₖ = NLPModels.obj(nlp, x)
	verbose && @printf "iter temps fₖ norm(gₖ,2) Δ ρₖ accuracy\n" 

	cgtol = one(T)  # Must be ≤ 1.
	cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
	ρₖ = -1

	# stop condition
	absolute(n,gₖ,ϵ) = nrm2(n,gₖ) > ϵ
	relative(n,gₖ,ϵ,∇fNorm2) = nrm2(n,gₖ) > ϵ * ∇fNorm2
	_max_iter(iter, max_iter) = iter < max_iter
	_max_time(start_time) = (time() - start_time) < max_time
	while absolute(n,gₖ,ϵ) && relative(n,gₖ,ϵ,∇fNorm2) && _max_iter(iter, max_iter) & _max_time(start_time) && isnan(ρₖ)==false# stop condition
		iter += 1
    fₖ = NLPModels.obj(nlp, x)
    NLPModels.grad!(nlp, x, gₖ)

		(verbose || data ) && (acc = KnetNLPModels.accuracy(nlp))
		verbose && @printf "%3d %4g %8.1e %7.1e %7.1e %7.1e %8.3e\n" iter (time() - start_time) fₖ norm(gₖ,2) Δ  ρₖ acc 
		data && push_acc!(nlp.counter, acc)
   
		cg_res = Krylov.cg(B, - gₖ, atol=T(atol), rtol=cgtol, radius = T(Δ), itmax=max(2 * n, 50))
		sₖ .= cg_res[1]  # result of the linear system solved by Krylov.cg
    
		(ρₖ, fₖ₊₁) = compute_ratio(x, fₖ, sₖ, nlp, B, gₖ) # we compute the ratio
		# step acceptance + update f,g
		if ρₖ > η
      xtmp .= x
			x .= x .+ sₖ
			epv_from_epv!(nlp.epv_work, nlp.epv_g)
			NLPModels.grad!(nlp, x, gₖ)
			minus_epv!(nlp.epv_work)
			add_epv!(nlp.epv_g, nlp.epv_work) # compute epv_y
			epv_from_v!(nlp.epv_s, sₖ)
			PartitionedStructures.update!(nlp.eplom_B, nlp.epv_work, nlp.epv_s; name=nlp.name)
			fₖ = fₖ₊₁
			@printf "✅\n"
		else
			@printf "❌\n"
		end				
		# now we update ∆
		(ρₖ >= η₁) && ((norm(sₖ, 2) > 0.8*Δ) ? Δ = ϕ*Δ : Δ = Δ)
		(ρₖ <= η) && (Δ = (1/ϕ)*Δ)
		
		# change the minibatch
		is_KnetNLPModel && reset_minibatch_train!(nlp)
		
	end
	@printf "iter temps fₖ norm(gₖ,2) Δ ρₖ accuracy\n" 
	@printf "%3d %4g %8.1e %7.1e %7.1e %7.1e %8.3e" iter (time() - start_time) fₖ norm(gₖ,2) Δ  ρₖ KnetNLPModels.accuracy(nlp)

	return (x, iter)
end

function compute_ratio(x::Vector{T}, fₖ::T, sₖ::Vector{T}, nlp::AbstractNLPModel, B::AbstractLinearOperator{T}, gₖ::AbstractVector{T}) where T <: Number
	mₖ₊₁ =  fₖ + dot(gₖ,sₖ) + 1/2 * (dot((B*sₖ),sₖ))
	xₖ₊₁ = x+sₖ
	fₖ₊₁ = NLPModels.obj(nlp, xₖ₊₁)
	set_vars!(nlp, x)
	ρₖ = (fₖ - fₖ₊₁)/(fₖ - mₖ₊₁)
	return (ρₖ,fₖ₊₁)
end