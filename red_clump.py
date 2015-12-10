from residuals import Sample,elems,doubleResidualHistPlot
from empca import empca
import numpy as np

genstep = {'specs':False,
		   'autopixplot':False,
		   'pixplot':True,
		   'pixfit':False,
		   'ransig':False,
		   'weight':False}

rcsample = Sample('red_clump',30,2,genstep,label = 'FE_H',up = -0.4,low=-0.5,fontsize = 10)
rcsample.getData()
rcsample.snrCut()
rcsample.maskData()
rcsample.snrCorrect()
rcsample.pixFit(100)
'''
rcsample.allPixFit()
rcsample.allErrPixFit()
rcsample.allRandomSigma()
weighted = np.zeros((len(elems),len(rcsample.specs)))
i=0
for elem in elems:
	weightedr = rcsample.weighting(rcsample.residual,elem,
								   rcsample.residElemName(elem))
	weighteds = rcsample.weighting(rcsample.sigma,elem,
								   rcsample.sigmaElemName(elem))
	doubleResidualHistPlot(elem,weightedr,weighteds,
						   rcsample.residhistElemPlotName(elem),
						   bins = 50)
	weighted[i] = weightedr
	i+=1

red_clump_model = empca(weighted)
'''