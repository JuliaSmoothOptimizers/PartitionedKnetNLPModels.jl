
@testset "ANN definition" begin
	DenseNet = Chainnll(Dense(784,50), Dense(50,10))
	LeNet = Chainnll(Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity))
	
	C = 10 # nombre de classe ≈ N 
	layer_PS = [40,20,1] 
	PSNet = Chainnll(Conv(4,4,1,20), Conv(4,4,20,50), SL(800,C,layer_PS[1]),  SL(C*layer_PS[1],C,layer_PS[2]), SL(C*layer_PS[2],C,layer_PS[3];f=identity))


	Ps_DenseNet = precompile_ps_struct(DenseNet)	
	Ps_LeNet = precompile_ps_struct(LeNet)
	Ps_PSNet = precompile_ps_struct(PSNet)

	psd = PS_deduction(Ps_DenseNet)
	psl = PS_deduction(Ps_LeNet)
	psp = PS_deduction(Ps_PSNet)

	@testset "test PS DenseNet" begin
		all_dep_psd = reduce(((x,y) ->  unique!(vcat(x,y))), psd)
		@test length(all_dep_psd) == length(vector_params(DenseNet))
		for i in 1:length(psd)
			@test length(psd[i]) == 39301 # == 784*50 (première couche dense) + 50 (biais) + 50 (deuxième couche dense) + 1 (biais 2e couche)
		end 		
	end 
	
end 


