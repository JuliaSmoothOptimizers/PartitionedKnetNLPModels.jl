mutable struct Counter_accuracy{T}
	acc :: Vector{T}	
end

Counter_accuracy(T :: DataType) = Counter_accuracy{T}(Vector{T}(undef,0))
push_acc!(counter :: Counter_accuracy{T}, value :: T) where T = push!(counter.acc, value :: T)