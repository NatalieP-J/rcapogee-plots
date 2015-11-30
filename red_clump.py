from residuals import Sample,elems,doubleResidualHistPlot

genstep = {'specs':False,
		   'autopixplot':True,
		   'pixplot':False,
		   'pixfit':True,
		   'ransig':True,
		   'weight':True}

rcsample = Sample('red_clump',30,2,genstep,label = 'FE_H',up = -0.2,low=-0.1,fontsize = 10)
rcsample.getData()
rcsample.snrCut()
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
