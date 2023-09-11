well_defined(β) = β > sqrt(eps(β))
armijo(fₖ, fₖ₊₁, β, t) = fₖ - fₖ₊₁ ≥ β *  t

"""
    basic_linesearch(x::Vector{T}, fₖ::T, sₖ::Vector{T}, nlp::AbstractNLPModel{T}, gₖ::AbstractVector{T}; vec=similar(x), c=0.05, τ=2/3, init=1., β = init, α=0.05 ) where T <: Number

Find a step length β satisfying the Armijo condition. β may be > 1.
"""
function basic_linesearch(x::Vector{T},
  fₖ::T,
  sₖ::Vector{T},
  nlp::AbstractNLPModel,
  gₖ::AbstractVector{T}; # -g
  vec=similar(x),
  c=0.05,
  τ=2/3,
  init=1.,
  β = init,
  α=0.05
) where T <: Number
  build_v!(nlp.epv_g)
  m = dot(gₖ, sₖ)
  t = c * m # dot(-g, s)

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
    # x .= vec
  else # set fₖ₊₁ to fₖ
    β = -1.
    @printf "❌ β too small! "
    fₖ₊₁ = fₖ
  end
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  # @printf "β = %8.3e ≥ 1, fₖ = %8.3e, fₖ₊₁ = %8.3e \n" β fₖ fₖ₊₁
	return β
end



"""
  backtracking_linesearch(x::Vector{T}, fₖ::T, sₖ::Vector{T}, nlp::AbstractNLPModel{T}, gₖ::AbstractVector{T}; vec=similar(x), c=0.05, τ=2/3, init=1., β = init, α=0.05 ) where T <: Number

Find a step length β satisfying the Armijo condition, β ≤ 1.
"""
function backtracking_linesearch(x::Vector{T},
  fₖ::T,
  sₖ::Vector{T},
  nlp::AbstractNLPModel,
  gₖ::AbstractVector{T}; # -g
  vec=similar(x),
  c=0.05,
  τ=2/3,
  init=1.,
  β = init,
  α=0.05
) where T <: Number
  build_v!(nlp.epv_g)
  m = dot(gₖ, sₖ)
  t = c * m # dot(-g, s)

  vec .= x .+ β .* sₖ # xₖ₊₁
  fₖ₊₁ = NLPModels.obj(nlp, vec; α)
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁

  while well_defined(β)
    vec .= x .+ β .* sₖ # xₖ₊₁
    fₖ₊₁ = NLPModels.obj(nlp, vec; α)
    armijo(fₖ, fₖ₊₁, β, t) ? break : β = τ * β
    # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  end

  if well_defined(β) # update x, fₖ₊₁ is up to date    
  else # set fₖ₊₁ to fₖ
    β = -1.
    @printf "❌ β too small! "
  end
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  # @printf "β = %8.3e ≥ 1, fₖ = %8.3e, fₖ₊₁ = %8.3e \n" β fₖ fₖ₊₁
	return β
end


"""
  backtracking_linesearch(x::Vector{T}, fₖ::T, sₖ::Vector{T}, nlp::AbstractNLPModel{T}, gₖ::AbstractVector{T}; vec=similar(x), c=0.05, τ=2/3, init=1., β = init, α=0.05 ) where T <: Number

Find a step length β satisfying the Armijo condition, β ≤ 1.
WIP
"""
function decaying_linesearch(x::Vector{T},
  fₖ::T,
  sₖ::Vector{T},
  nlp::AbstractNLPModel,
  gₖ::AbstractVector{T}; # -g
  vec=similar(x),
  c=0.05,
  τ=2/3,
  init=1.,
  β = init,
  α=0.05
) where T <: Number
  build_v!(nlp.epv_g)
  m = dot(gₖ, sₖ)
  t = c * m # dot(-g, s)

  vec .= x .+ β .* sₖ # xₖ₊₁
  fₖ₊₁ = NLPModels.obj(nlp, vec; α)
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁

  while well_defined(β)
    vec .= x .+ β .* sₖ # xₖ₊₁
    fₖ₊₁ = NLPModels.obj(nlp, vec; α)
    armijo(fₖ, fₖ₊₁, β, t) ? break : β = τ * β
    # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  end

  if well_defined(β) # update x, fₖ₊₁ is up to date    
  else # set fₖ₊₁ to fₖ
    β = -1.
    @printf "❌ β too small! "
  end
  # @printf "β = %8.3e, ||β.*sₖ||= %8.3e, fₖ₊₁= %8.3e\n" β norm(β .* sₖ) fₖ₊₁
  # @printf "β = %8.3e ≥ 1, fₖ = %8.3e, fₖ₊₁ = %8.3e \n" β fₖ fₖ₊₁
	return β
end
