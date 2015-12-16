from residuals import Sample,elems,doubleResidualHistPlot
import numpy as np
import access_spectrum as acs
from apogee.tools import bitmask

genstep = {'specs':False,
		   'remask':True,
		   'autopixplot':False,
		   'pixplot':False,
		   'pixfit':True,
		   'ransig':True,
		   'weight':True}

clusterlis = ['M107','M13','M15','M2','M3','M5','M53','M71','M92','N5466','M67','N188','N2158','N2420','N4147','N6791','N6819','N7789']

GCs = ['M107','M13','M15','M2','M3','M5','M53','M71','M92','N5466']
OCs = ['M67','N188','N2158','N2420','N4147','N6791','N6819','N7789']

clusters = Sample('clusters',15,2,genstep=genstep,fontsize = 10)
clusters.getData()
clusters.snrCut()
clusters.maskData(maskvals = [0,1,2,3,4,5,6,7,12])
clusters.snrCorrect()
masterdata = np.copy(clusters.data)
masterspecs = np.copy(clusters.specs)
mastererrs = np.copy(clusters.errs)
mastermask = np.copy(clusters.mask)
masterbitmask = np.copy(clusters.bitmask)
OCres = {}
OCsig = {}
OCmembers = {}
OC_allres = np.zeros((87,7214))
OC_allsigs = np.zeros((87,7214))
OC_allspecs = np.zeros((87,7214))
OC_allmasks = np.zeros((87,7214))
OC_allbitmasks = np.zeros((87,7214),dtype=int)
i = 0
for c in OCs:
	match = np.where(masterdata['CLUSTER'] == c)
	clusters.data = clusters.data[match]
	nummembers = len(clusters.data)
	if c in OCs:
		OCmembers[c] = nummembers
	clusters.specs = clusters.specs[match]
	clusters.errs = clusters.errs[match]
	clusters.mask = clusters.mask[match]
	clusters.bitmask = clusters.bitmask[match]
	clusters.allPixFit(cluster=c)
	clusters.allRandomSigma(cluster=c)
	if c in OCs and nummembers > 9:
		for star in range(len(clusters.residual[0])):
			OC_allres[i] = clusters.residual[:,star]
			OC_allsigs[i] = clusters.sigma[:,star]
			OC_allspecs[i] = clusters.specs[star]
			OC_allmasks[i] = clusters.mask[star]
			OC_allbitmasks[i] = clusters.bitmask[star]
			i+=1
	for elem in elems:

		weightedr = clusters.weighting(clusters.residual,elem,
									  clusters.residElemName(elem,cluster=c))
		weighteds = clusters.weighting(clusters.sigma,elem,
									  clusters.sigmaElemName(elem,cluster=c))
		#doubleResidualHistPlot(elem,weightedr,weighteds,
		#					   clusters.residhistElemPlotName(elem,cluster=c),
		#					   bins = 50)
		if elem not in OCsig.keys():
			OCsig[elem] = []
		if elem not in OCres.keys():
			OCres[elem] = []
		if c in OCs and nummembers > 9:
			OCsig[elem].append(weighteds)
			OCres[elem].append(weightedr)
	clusters.data = masterdata
	clusters.specs = masterspecs
	clusters.errs = mastererrs
	mastermask[match] = clusters.mask
	clusters.mask = mastermask
	clusters.bitmask = masterbitmask
clusters.saveFiles()

'''
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

mask = np.where(OC_allmasks.T==0)
nomask = np.where(OC_allmasks.T != 0)

OCresmask = np.zeros(OC_allres.T.shape)
OCresmask[mask] = True
OCresmask[nomask] = False

OCresmasked = np.ma.masked_array((OC_allres).T,mask = OCresmask)

covar = np.ma.cov(OCresmasked)
plt.figure(figsize = (12,10))
plt.imshow(covar,vmin = -1e-4,vmax =1e-4,interpolation = 'nearest',cmap = 'viridis')
plt.colorbar()
plt.savefig('PixelSpaceCovariance.png')
plt.close()
# How does np.ma.cov decide what values to mask?

for elem in elems:
	OCres[elem] = np.array([item for sublist in OCres[elem] for item in sublist])
	OCsig[elem] = np.array([item for sublist in OCsig[elem] for item in sublist])
	#doubleResidualHistPlot(elem,OCres[elem],OCsig[elem],'./OC_{0}combine.png'.format(elem))

elemarray = np.array(OCres.values())


plt.figure(figsize = (12,10))
plt.imshow(np.cov(elemarray),vmin = -1e-4,vmax =1e-4,interpolation = 'nearest',cmap = 'viridis')
for i in range(len(elems)):
	plt.text(0,i,elems[i],color='black')
for i in range(1,len(elems)):
	plt.text(i,0,elems[i],color='black')
plt.colorbar()
plt.savefig('ElementSpaceCovariance.png')
plt.close()


plt.figure(1,figsize=(16,14))
OC_allresplot = np.copy(OC_allres)
mask = np.where(OC_allmasks==0)
OC_allresplot[mask] = np.nan
plt.imshow(abs(OC_allresplot/OC_allsigs),vmax = 1.1,aspect = 100,interpolation='nearest',cmap = 'viridis')
plt.figtext(0.1,0.2,'M67')
plt.axhline(24,linewidth = 2,color='r')
plt.figtext(0.1,0.35,'N2158')
plt.axhline(24+10,linewidth = 2,color='r')
plt.figtext(0.1,0.53,'N6791')
plt.axhline(24+10+23,linewidth = 2,color='r')
plt.figtext(0.1,0.77,'N6819')
plt.ylim(0,87)
plt.colorbar()
plt.savefig('NormalizedOCResiduals.png')
plt.close()

plt.figure(5,figsize=(16,14))
plt.imshow(abs(OC_allresplot),aspect = 100,vmax = 0.06,interpolation='nearest',cmap = 'viridis')
plt.figtext(0.1,0.2,'M67')
plt.axhline(24,linewidth = 2,color='r')
plt.figtext(0.1,0.35,'N2158')
plt.axhline(24+10,linewidth = 2,color='r')
plt.figtext(0.1,0.53,'N6791')
plt.axhline(24+10+23,linewidth = 2,color='r')
plt.figtext(0.1,0.77,'N6819')
plt.ylim(0,87)
plt.colorbar()
plt.savefig('OCResiduals.png')
plt.close()

plt.figure(2,figsize=(16,14))
OC_allspecsplot = np.copy(OC_allspecs)
plt.imshow(abs(OC_allspecsplot),aspect = 100,interpolation='nearest',cmap = 'viridis')
plt.figtext(0.1,0.2,'M67')
plt.axhline(24,linewidth = 2,color='r')
plt.figtext(0.1,0.35,'N2158')
plt.axhline(24+10,linewidth = 2,color='r')
plt.figtext(0.1,0.53,'N6791')
plt.axhline(24+10+23,linewidth = 2,color='r')
plt.figtext(0.1,0.77,'N6819')
plt.ylim(0,87)
plt.colorbar()
plt.savefig('OCspectra.png')
plt.close()

plt.figure(3,figsize=(16,14))
OC_allmasksplot = np.copy(OC_allmasks)
OC_allmasksplot[np.where(OC_allmasks!=0)] = np.nan
plt.imshow(OC_allmasksplot,aspect = 100,interpolation='nearest',cmap = 'viridis')
plt.figtext(0.1,0.2,'M67')
plt.axhline(24,linewidth = 2,color='r')
plt.figtext(0.1,0.35,'N2158')
plt.axhline(24+10,linewidth = 2,color='r')
plt.figtext(0.1,0.53,'N6791')
plt.axhline(24+10+23,linewidth = 2,color='r')
plt.figtext(0.1,0.77,'N6819')
plt.ylim(0,87)
plt.colorbar()
plt.savefig('OCmask.png')
plt.close()

plt.figure(4,figsize=(16,14))
OC_allbitmasksplot = np.copy(OC_allbitmasks).astype(np.float64)
OC_allbitmasksplot[np.where(OC_allbitmasks==0)] = np.nan
plt.imshow(np.log2(OC_allbitmasksplot),aspect = 100,interpolation='nearest',cmap = 'viridis')
plt.figtext(0.1,0.2,'M67')
plt.axhline(24,linewidth = 2,color='r')
plt.figtext(0.1,0.35,'N2158')
plt.axhline(24+10,linewidth = 2,color='r')
plt.figtext(0.1,0.53,'N6791')
plt.axhline(24+10+23,linewidth = 2,color='r')
plt.figtext(0.1,0.77,'N6819')
plt.ylim(0,87)
plt.colorbar()
plt.savefig('OCbitmask.png')
plt.close()
'''

