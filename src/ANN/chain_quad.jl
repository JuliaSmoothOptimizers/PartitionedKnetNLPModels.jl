mutable struct Chain_psloss_quad1 <: KnetNLPModels.Chain
	layers
	Chain_psloss_quad1(layers...) = new(layers)
end
(c::Chain_psloss_quad1)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_psloss_quad1)(x,y) = myloss_quad1(c(x),y)
(c::Chain_psloss_quad1)(d::Knet.Data) = myloss_quad1(c; data=d, average=true)
function myloss_quad1(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = myloss_quad1(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function myloss_quad1(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = (x->x^2).(scores .- maximum(scores, dims=1)) # enlever cette ligne fonctionne moins bien fonction	
	acc = sum(scores[indices])
	average ? (acc / length(labels)) : (acc, length(labels))
end


mutable struct Chain_psloss_quad2 <: KnetNLPModels.Chain
	layers
	Chain_psloss_quad2(layers...) = new(layers)
end
(c::Chain_psloss_quad2)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_psloss_quad2)(x,y) = myloss_quad2(c(x),y)
(c::Chain_psloss_quad2)(d::Knet.Data) = myloss_quad2(c; data=d, average=true)
function myloss_quad2(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = myloss_quad2(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function myloss_quad2(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = scores .- maximum(scores, dims=1) # enlever cette ligne fonctionne moins bien fonction	
	acc = sum((x->x^2).(scores[indices]))
	ninstances = length(labels); y1 = size(scores,1) 
	tmp = [1:(ninstances*y1);]; splice!(tmp, indices)
	acc += sum(scores[tmp])
	average ? (acc / length(labels)) : (acc, length(labels))
end