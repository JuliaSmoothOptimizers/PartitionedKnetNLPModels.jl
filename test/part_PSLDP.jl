
@testset "PartitionedKnetNLPModel and PUS tests" begin
  C = 10 # nombre de classe ≈ N 
  layer_PS = [20,10,1] 
  Part_PSNet = PartChainPSLDP(Conv(4,4,1,6), Conv(4,4,6,16), SL(160,C,layer_PS[1]),  SL(C*layer_PS[1],C,layer_PS[2]), SL(C*layer_PS[2],C,layer_PS[3];f=identity))

  (xtrn, ytrn) = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
  (xtst, ytst) = MNIST.testdata(Float32); ytst[ytst.==0] .= 10

  size_minbatch = 100
  dtrn = create_minibatch(xtrn, ytrn, size_minbatch)	 	 
  dtst = create_minibatch(xtst, ytst, size_minbatch)

  max_time = 25.
  max_iter = 100

  nlp_plbfgs = PartitionedKnetNLPModel(Part_PSNet; name=:plbfgs)
  ges_plbfgs = PUS(nlp_plbfgs; max_time, max_iter) #max_iter=10000 = 100 epochs
end