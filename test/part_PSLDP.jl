

C = 10 # nombre de classe ≈ N 
layer_PS = [40,20,1] 
PSNet = PartChainPSLDP(Conv(4,4,1,20), Conv(4,4,20,50), SL(800,C,layer_PS[1]),  SL(C*layer_PS[1],C,layer_PS[2]), SL(C*layer_PS[2],C,layer_PS[3];f=identity))

(xtrn, ytrn) = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
(xtst, ytst) = MNIST.testdata(Float32); ytst[ytst.==0] .= 10

size_minbatch = 100
dtrn = create_minibatch(xtrn, ytrn, size_minbatch)	 	 
dtst = create_minibatch(xtst, ytst, size_minbatch)
