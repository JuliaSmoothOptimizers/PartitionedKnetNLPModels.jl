function findindices(scores, labels::AbstractArray{<:Integer}; dims=1)
	ninstances = length(labels)
	nindices = 0
	indices = Vector{Int}(undef,ninstances)
	if dims == 1                   # instances in first dimension
			y1 = size(scores,1)
			y2 = div(length(scores),y1)
			if ninstances != y2; throw(DimensionMismatch()); end
			@inbounds for j=1:ninstances
					if labels[j] == 0; continue; end
					indices[nindices+=1] = (j-1)*y1 + labels[j]
			end
	elseif dims == 2               # instances in last dimension
			y2 = size(scores,ndims(scores))
			y1 = div(length(scores),y2)
			if ninstances != y1; throw(DimensionMismatch()); end
			@inbounds for j=1:ninstances
					if labels[j] == 0; continue; end
					indices[nindices+=1] = (labels[j]-1)*y1 + j
			end
	else
			error("findindices only supports dims = 1 or 2")
	end
	return (nindices == ninstances ? indices : view(indices,1:nindices))
end