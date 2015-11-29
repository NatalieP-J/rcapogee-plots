from residuals import Sample,elems,doubleResidualHistPlot

genstep = {'specs':True,
		   'autopixplot':False,
		   'pixplot':False,
		   'pixfit':True,
		   'ransig':True,
		   'weight':True}

rcsample = Sample('red_clump',15,2,genstep,label = 'FE_H',low = -0.5,up = -0.4,fontsize = 10)
rcsample.getData()
rcsample.maskData()
#rcsample.snrCorrect()
rcsample.allPixFit()
rcsample.allRandomSigma()
for elem in elems:
	weightedr = rcsample.weighting(rcsample.residual,elem,
								   rcsample.residElemName(elem))
	weighteds = rcsample.weighting(rcsample.sigma,elem,
								   rcsample.sigmaElemName(elem))
	doubleResidualHistPlot(elem,weightedr,weighteds,
						   rcsample.residhistElemPlotName(elem),
						   bins = 50)
