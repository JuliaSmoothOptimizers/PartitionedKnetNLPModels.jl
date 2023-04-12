"""
   	f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(nlp::PartitionedKnetNLPModel{T, S, C}, w::AbstractVector{T}) where {T, S, C}
  increment!(nlp, :neval_obj)
  set_vars!(nlp, w)
  res = nlp.chain(nlp.current_training_minibatch)
  f_w = sum(sum.(res))
  return f_w
end

"""
		g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  nlp::PartitionedKnetNLPModel{T, S, C},
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S, C}
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_grad)
  set_vars!(nlp, w)
  partitioned_gradient!(nlp)
  build_v!(nlp.epv_g)
  g .= get_v(nlp.epv_g)
  return g
end
