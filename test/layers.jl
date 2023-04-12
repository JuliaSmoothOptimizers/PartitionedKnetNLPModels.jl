
@testset "ANN definition" begin
  DenseNet = Chain_NLL(Dense(784, 50), Dense(50, 10))
  LeNet =
    Chain_NLL(Conv(5, 5, 1, 20), Conv(5, 5, 20, 50), Dense(800, 500), Dense(500, 10, identity))

  C = 10 # nombre de classe ≈ N 
  layer_PS = [40, 20, 1]
  PSNet = Chain_NLL(
    Conv(4, 4, 1, 20),
    Conv(4, 4, 20, 50),
    SL(800, C, layer_PS[1]),
    SL(C * layer_PS[1], C, layer_PS[2]),
    SL(C * layer_PS[2], C, layer_PS[3]; f = identity),
  )

  Ps_DenseNet = precompile_ps_struct(DenseNet)
  Ps_LeNet = precompile_ps_struct(LeNet)
  Ps_PSNet = precompile_ps_struct(PSNet)

  PS_deduction = PartitionedKnetNLPModels.PS_deduction
  psd = PS_deduction(Ps_DenseNet)
  psl = PS_deduction(Ps_LeNet)
  psp = PS_deduction(Ps_PSNet)

  vector_params = KnetNLPModels.vector_params
  @testset "test PS DenseNet" begin
    all_dep_psd = reduce(((x, y) -> unique!(vcat(x, y))), psd)
    @test length(all_dep_psd) == length(vector_params(DenseNet))

    for i = 1:length(psd)
      @test length(psd[i]) == 39301 # == 784*50 (première couche dense) + 50 (biais) + 50 (deuxième couche dense) + 1 (biais 2e couche)
    end

    for i = 1:length(psd)
      @test psd[1][1:39000] == psd[i][1:39000]
    end
  end

  @testset "test PS LeNet" begin
    all_dep_psl = reduce(((x, y) -> unique!(vcat(x, y))), psl)
    @test length(all_dep_psl) == length(vector_params(LeNet))

    for i = 1:length(psl)
      @test length(psl[i]) == 426571
    end

    for i = 1:length(psl)
      @test psl[1][1:426000] == psl[i][1:426000]
    end
  end

  @testset "test PS PsNet" begin
    all_dep_psp = reduce(((x, y) -> unique!(vcat(x, y))), psp)
    @test length(all_dep_psp) == length(vector_params(PSNet))

    for i = 1:length(psp)
      @test length(psp[i]) == 6026
    end

    for i = 1:length(psp)
      @test psp[1][1:300] == psp[i][1:300]
    end
  end
end
