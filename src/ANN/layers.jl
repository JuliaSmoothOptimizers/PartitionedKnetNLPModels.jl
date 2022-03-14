using Knet

#= 
--------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------- Couche de convolution ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------
=# 

struct PsConv; w; b; dp; lengthlayer::Int end 
# Peut importe les dépendances initiales, ie: qu'elles soient larges ou petites elles seront toutes transmises à la couce suivante.
# Seules les variabes propres au filtrage sont conservées et distinctes.
PsConv(w1,w2,cx,cy; index=0) = PsConv([index + (j-1)*w2 + (p-1)*w1*w2 + (s-1)*w1*w2*cx + i for i=1:w1,j=1:w2,p=1:cx,s=1:cy], [index+w1*w2*cx*cy+ i for i=1:cy], ones(Bool,w1,w2,cx,cy), w1*w2*cx*cy+cy)
size(psc::PsConv) = size(psc.w)
input(psc::PsConv) = maximum(size(psc)) # non claire dans le cas de Conv, en pratique doit être >0

function ps(psc::PsConv, dep::Vector{Vector{Int}}; dp=ones(Bool,pss.in))
	common_dep = reduce(((x,y)->unique!(vcat(x,y))), dep; init=[])
	(w1,w3,cx,cy) = size(psc)
	new_dep = Vector{Vector{Int}}(undef,cy)
	for i in 1:cy
		tmp_dep = vec(psc.w[:,:,:,i])
		total_tmp_dep = vcat(common_dep, tmp_dep, psc.b[i])
		new_dep[i] = total_tmp_dep
	end 
	return new_dep
end 

struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
# (c::Conv)(x) = begin r = c.f.(pool(conv4(c.w, x) .+ c.b)); @show typeof(r), size(r); r end 
Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)
ps_struct(c::Conv; index::Int=0) = PsConv(size(c.w)...; index=index)


#= 
--------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------- Couche dense ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------
=# 

struct PsDense; var_layer::Array{Int,2}; bias_layer::Vector{Int}; dp::Vector{Bool}; lengthlayer::Int end #total length #var_layer+#bias_layer
PsDense(in::Int,out::Int; index::Int=0) = PsDense( [(index + (j-1)*out + i) for i=1:out, j=1:in], [(index + in*out + i) for i=1:out], trues(in), (in+1)*out) # in+1 : +1 pour considérer le biais
size(d::PsDense) = size(d.var_layer)
input(d::PsDense) = size(d)[2]

"""
    ps(psd::PsDense, Dep::Vector{Vector{int}})
pour chaque noeud on va ajouter les dépendances du layer propre (excepté dropout) 
on y ajoute ensuite les dépendances (car dense) de la couche précédente modélisé par Di (à modérer avec le dropout)
Le cas de la couche Dense est particulier, sans dropout il n'a que peu d'intérêt
"""
function ps(psd::PsDense, dep::Vector{Vector{Int}}; dp=ones(Bool,pss.in))
	(out,in) = size(psd)
	length(dp)==in && psd.dp .= dp
	if length(dep)==in
		transformed_dep =dep
	elseif in%length(dep)==0 #pour faire le lien avec les couches de convolution
		gcd = in/length(dep)
		transformed_dep= [dep[i] for j=1:gcd,i=1:length(dep)]
	else
		@error("nombre de dépendances et nombre de neurones distincts PsDense")	
	end 
	new_dep = Vector{Vector{Int}}(undef, out)
	for i in 1:out
		pertinent_indices = findall(psd.dp) #obtention des indices du dropout
		interlayer_dep = vcat(psd.var_layer[i,pertinent_indices], psd.bias_layer[i]) # dépendances du layer propre
		# accumulation_dep = dep[pertinent_indices] # les dépendances de la couche précédente propagée		
		accumulation_dep = vcat(transformed_dep[pertinent_indices]...) # les dépendances de la couche précédente propagée
		new_dep[i] = sort!(unique!(vcat(interlayer_dep, accumulation_dep))) # concaténation des dépendances
	end 
	return new_dep	
end 

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b)
size(d::Dense) = size(d.w)
# (d::Dense)(x) = begin @show size(x); r=d.f.(d.w * mat(dropout(x,d.p)) .+ d.b); @show typeof(r), size(r); r end # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=sigm;pdrop=0.) = Dense(param(o,i), param0(o), f, pdrop)
ps_struct(d::Dense; index::Int=0) = begin (o,i)=size(d.w); PsDense(i,o; index=index) end

checkfloor(N,elt) = (Int)(max(floor(elt/N), 1)) 
checkfloor(N,e1,e2) = Vector{Int}([checkfloor(N,e1), checkfloor(N,e2)])

#= 
--------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------- Couche séparable ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------
=# 

struct PsSep; vec_wi::Vector{Array{Int,2}}; vec_bi::Vector{Vector{Int}}; dp::Vector{Bool}; N; gcdi; gcdo; lengthlayer::Int; in; out end #total length #var_layer+#bias_layer
size(pss::PsSep) = (pss.out,pss.in,pss.N)
input(pss::PsSep) = pss.in
function PsSep(in::Int,out::Int; index::Int=0, N::Int=gcd(in,out))
	(gcdi,gcdo) = checkfloor(N,in), checkfloor(N,out)
	vec_wi = map( (j->zeros(Int32,gcdo,gcdi)) ,[1:N;])
	vec_bi = map( (x->zeros(Int32,gcdo)),[1:N;])
	for l in 1:N 		
		vec_wi[l] = [(index + (l-1)*gcdi*gcdo + (j-1)*gcdo + i) for i=1:gcdo, j=1:gcdi]
		vec_bi[l] = [(index + N*gcdi*gcdo + (l-1)*gcdo + i) for i=1:gcdo]
	end 	
	dp = trues(in)
	lengthlayer = N*gcdi*gcdo + N*gcdo
	return PsSep(vec_wi, vec_bi, dp, N, gcdi, gcdo, lengthlayer, in, out)	
end 

"""
    ps(pss::PsSep, Dep::Vector{Vector{int}})
pour chaque noeud on va ajouter les dépendances du layer propre: une sous partie de la couche précédente (+ dropout) 
on y ajoute ensuite certaines dépendances de la couche précédente modélisé par Di (à modérer avec le dropout)
Dans le cas PS, la propagation des variables sera moins intense que le cas Dense
"""
function ps(pss::PsSep, dep::Vector{Vector{Int}}; dp=ones(Bool,pss.in))
	(gcdo, gcdi, N, in) = (pss.gcdo, pss.gcdi, pss.N, pss.in)
	length(dp)==in && pss.dp .= dp
	l_dep = length(dep)
	if length(dep)==in
		transformed_dep =dep
	elseif in%length(dep)==0 #pour faire le lien avec les couches de convolution
		gcd = in/length(dep)
		transformed_dep= [dep[i] for j=1:gcd,i=1:length(dep)]
	else
		error("nombre de dépendances et nombre de neurones distincts PsSep, (in,dep)=$in,$l_dep")	
	end 
	new_dep = Vector{Vector{Int}}(undef, pss.out)	
	for i in 1:N
		brut_pertinent_indices = findall(pss.dp[((i-1)*gcdi+1):(i*gcdi)]) #obtention des indices du dropout sur la partie atteinte par la couche séparable
		pertinent_indices = brut_pertinent_indices .+ (i-1)*gcdi			
		accumulation_dep = vcat(transformed_dep[pertinent_indices]...) # les dépendances de la couche précédente propagée
		for j in 1:gcdo
			interlayer_dep = vcat(pss.vec_wi[i][j,brut_pertinent_indices], pss.vec_bi[i][j]) # dépendances du layer propre en prenant en compte le dropout
			new_dep[(i-1)*gcdo + j] = sort!(unique!(vcat(interlayer_dep, accumulation_dep))) # concaténation des dépendances
		end
	end 
	return new_dep	
end 

struct Sep_layer; vec_wi; vec_bi; N; f; gcdi; gcdo; in; out; p; end
(psd::Sep_layer)(x) = mapreduce( (j -> psd.f.( psd.vec_wi[j]*mat(dropout(x,psd.p))[(psd.gcdi*(j-1)+1):(psd.gcdi*j),:] .+ (psd.vec_bi[j]) ) ), ((x,y)->vcat(x,y)), [1:psd.N;] )
Sep_layer(i::Int,o::Int; N::Int=gcd(i,o),f=sigm,p=0.3) = Sep_layer( map((j->param(checkfloor(N,o,i)...)),[1:N;]), map((x->param0(checkfloor(N,o))),[1:N;]), N, f , checkfloor(N,i), checkfloor(N,o), checkfloor(N,i)*N, checkfloor(N,o)*N, p)
SL(i::Int,N::Int,ni::Int; f=sigm) = Sep_layer(i,N*ni;N=N,f=f)
ps_struct(psd::Sep_layer; index::Int=0) = PsSep(psd.in,psd.out; N=psd.N, index=index)

#= 
--------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------- Couche séparable 2 ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------
=# 

# struct Sep_layer2; mat_wi; bi; N; f; gcdi; gcdo; in; out; p; end
# (psd::Sep_layer2)(x) = begin @show "là"; outputs = param(psd.out,last(size(x))).value; mapreduce( (j -> view(outputs,(psd.gcdo*(j-1)+1):(psd.gcdo*j),:) .= psd.f.( view(psd.mat_wi,(psd.gcdo*(j-1)+1):(psd.gcdo*j),:) * view(Knet.mat(dropout(x,psd.p)),(psd.gcdi*(j-1)+1):(psd.gcdi*j),:) .+ view(psd.bi,(psd.gcdo*(j-1)+1):(psd.gcdo*j)) ) ), ((x,y)->vcat(x,y)), [1:psd.N;] ); outputs end 
# Sep_layer2(i::Int,o::Int; N::Int=gcd(i,o),f=sigm,p=0.3) = Sep_layer2( param(checkfloor(N,o)*N,checkfloor(N,i)), param0(checkfloor(N,o)*N), N, f , checkfloor(N,i), checkfloor(N,o), checkfloor(N,i)*N, checkfloor(N,o)*N, p)
# SL2(i::Int,N::Int,ni::Int; f=sigm) = Sep_layer2(i,N*ni;N=N,f=f)
# ps_struct(psd::Sep_layer2; index::Int=0) = PsSep(psd.in,psd.out; N=psd.N, index=index)
