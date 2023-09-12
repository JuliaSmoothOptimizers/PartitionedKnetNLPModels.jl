using CUDA 
@testset "PartitionedKnetNLPModel and PUS tests" begin
  ENV["DATADEPS_ALWAYS_ACCEPT"] = true # download datasets without having to manually confirm the download
  CUDA.allowscalar(true)

  (xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
  (xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
  data_train = (xtrn, ytrn)
  data_test = (xtst, ytst)

  C = 10
  layer_PS = [24,15,1] 

  size_minibatch = 100
  Part_PSNet = PartChainPSLDP(Conv(5,5,1,40; pool_option=1), Conv(5,5,40,30; pool_option=1), SL(480, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))

  max_time = 180.
  max_iter = 100

  # nlp_plbfgs = PartitionedKnetNLPModel(Part_PSNet; name=:plbfgs, data_train, data_test, size_minibatch)  
  # ges_plbfgs = PUS(nlp_plbfgs; max_time, max_iter) #max_iter=10000 = 100 epochs

  # nlp_plbfgs = PartitionedKnetNLPModel(Part_PSNet; name=:plbfgs, data_train, data_test, size_minibatch)
  # ges_lbfgs = LBFGS(nlp_lbfgs; max_time, max_iter) 

  printing=false
  α=0.
  mem=10
  nlp_plbfgs = PartitionedKnetNLPModel(Part_PSNet; name=:plbfgs, data_train, data_test, size_minibatch, mem)
  ges_plse100 = PLS(nlp_plbfgs; max_time, max_iter, printing, α)
end