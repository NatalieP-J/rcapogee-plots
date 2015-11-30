from residuals import Sample,elems,doubleResidualHistPlot
import numpy as np

genstep = {'specs':False,
		   'autopixplot':True,
		   'pixplot':False,
		   'pixfit':True,
		   'ransig':True,
		   'weight':True}

clusterlis = ['M107','M13','M15','M2','M3','M5','M53','M71','M92','N5466','M67','N188','N2158','N2420','N4147','N6791','N6819','N7789']

clusters = Sample('clusters',15,2,genstep,fontsize = 10)
clusters.getData()
clusters.maskData()
clusters.snrCorrect()
masterdata = np.copy(clusters.data)
masterspecs = np.copy(clusters.specs)
mastererrs = np.copy(clusters.errs)
for c in clusterlis:
	match = np.where(masterdata['CLUSTER'] == c)
	clusters.data = clusters.data[match]
	clusters.specs = clusters.specs[match]
	clusters.errs = clusters.errs[match]
	clusters.allPixFit(cluster=c)
	clusters.allRandomSigma(cluster=c)
	for elem in elems:
		weightedr = clusters.weighting(clusters.residual,elem,
									  clusters.residElemName(elem,cluster=c))
		weighteds = clusters.weighting(clusters.sigma,elem,
									  clusters.sigmaElemName(elem,cluster=c))
		doubleResidualHistPlot(elem,weightedr,weighteds,
							   clusters.residhistElemPlotName(elem,cluster=c),
							   bins = 50)
	clusters.data = masterdata
	clusters.specs = masterspecs
	clusters.errs = mastererrs
