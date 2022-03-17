using LinearOperators, NLPModels, LinearAlgebra, LinearAlgebra.BLAS, Krylov
using Printf, SolverTools, SolverCore


mapnan(v::AbstractVector) = mapreduce((vi-> isnan(vi)), |, v)

LBFGS(nlp :: AbstractNLPModel; kwargs...) = Generic_LBFGS(nlp;kwargs...)
LBFGS(nlp :: KnetNLPModel; kwargs...) = Generic_LBFGS(nlp; is_KnetNLPModel=true, kwargs...)
function Generic_LBFGS(nlp :: AbstractNLPModel;
	x::AbstractVector=copy(nlp.meta.x0),
	T::DataType = eltype(x),
	kwargs...)
	B = LBFGSOperator(T,nlp.meta.nvar, scaling=true) #:: LBFGSOperator{T} #scaling=true
	println("LBFGSOperator{$T} ✅")
	return Generic_LBFGS(nlp, B; x=x, kwargs...)
end

function Generic_LBFGS(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
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

	println("Début LBFGS TR CG")
	(x,iter) = TR_CG_ANLP_LO(nlp, B; max_eval=max_eval, max_time=max_time, kwargs...)

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


function TR_CG_ANLP_LO(nlp :: AbstractNLPModel, B :: AbstractLinearOperator{T};
	x::AbstractVector=copy(nlp.meta.x0),
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
	kwargs...,
	) where T <: Number

	iter = 0 # ≈ k
	gₖ = Vector{T}(undef,n); gₖ = ∇f₀
	gₜₘₚ = similar(gₖ)
	∇fNorm2 = nrm2(n, ∇f₀)
	Δₘₐₓ = Δ * ϕ^4

	fₖ = NLPModels.obj(nlp, x)
	
	@printf "iter temps fₖ norm(gₖ,2) Δ Δₘₐₓ ρₖ\n" 

	cgtol = one(T)  # Must be ≤ 1.
	cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
	ρₖ = -1

	# stop condition
	absolute(n,gₖ,ϵ) = nrm2(n,gₖ) > ϵ
	relative(n,gₖ,ϵ,∇fNorm2) = nrm2(n,gₖ) > ϵ * ∇fNorm2
	_max_iter(iter, max_iter) = iter < max_iter
	_max_time(start_time) = (time() - start_time) < max_time
	while absolute(n,gₖ,ϵ) && relative(n,gₖ,ϵ,∇fNorm2) && _max_iter(iter, max_iter) & _max_time(start_time) && isnan(ρₖ)==false# stop condition
		@printf "%3d %4g %8.1e %7.1e %7.1e %7.1e " iter (time() - start_time) fₖ norm(gₖ,2) Δ  ρₖ
		mod(iter,5) == 0 && @printf "\tCurrent accuracy: %8.3e " accuracy(nlp)

		iter += 1
		
		cg_res = Krylov.cg(B, - gₖ, atol=T(atol), rtol=cgtol, radius = T(Δ), itmax=max(2 * n, 50))
		sₖ = cg_res[1]  # result of the linear system solved by Krylov.cg


		if iszero(sₖ) # numerical unstability of LBFGS
			B = LBFGSOperator(T,nlp.meta.nvar, scaling=true)
			Δ = min(Δₘₐₓ,1/ϕ*Δ)
			@printf "norm(s= £8.1e❌❌\n" nrm2(sₖ)
		else # usual case 
			(ρₖ, fₖ₊₁) = compute_ratio(x, fₖ, sₖ, nlp, B, gₖ) # we compute the ratio
			# step acceptance + update f,g
			if ρₖ > η
				x .= x + sₖ; gₜₘₚ .= gₖ
				NLPModels.grad!(nlp, x, gₖ); fₖ = fₖ₊₁
				yₖ = gₖ - gₜₘₚ; push!(B, sₖ, yₖ)
				@printf "✅\n"
			else
				@printf "❌\n"
			end				
			# now we update ∆
			Δₘₐₓ = Δₘₐₓ*δ
			(ρₖ >= η₁ && norm(sₖ, 2) > 0.8*Δ) ? Δ = min(Δₘₐₓ,ϕ*Δ) : Δ = min(Δₘₐₓ,Δ)
			(ρₖ <= η) && (Δ = min(Δₘₐₓ,1/ϕ*Δ))
		end 
		
		# on change le minibatch
		is_KnetNLPModel && reset_minibatch_train!(nlp)
		
		# periodic printer
	end
	@printf "%3d %4g %8.1e %7.1e %7.1e %7.1e\n" iter (time() - start_time) fₖ norm(gₖ,2) Δ ρₖ
	@printf "Current accuracy: %8.1e \% " accuracy(nlp)

	return (x, iter)
end

function compute_ratio(x::AbstractVector{T}, fₖ::T, sₖ::Vector{T}, nlp::AbstractNLPModel, B::AbstractLinearOperator{T}, gₖ::AbstractVector{T}) where T <: Number
	mₖ₊₁ =  fₖ + dot(gₖ,sₖ) + 1/2 * (dot((B*sₖ),sₖ))
	fₖ₊₁ = NLPModels.obj(nlp, x+sₖ)
	ρₖ = (fₖ - fₖ₊₁)/(fₖ - mₖ₊₁)
	isnan(ρₖ) && @show mₖ₊₁, fₖ₊₁, fₖ, norm(sₖ,2)
	return (ρₖ,fₖ₊₁)
end


	

epv_tmp = epv_g
epv_g = partitioned_gradient()

add_epv!(epvg, minus_epv!(epv_tmp)) # compute epv_y