from residuals import Sample,elems,doubleResidualHistPlot,badcombpixmask
from empca import empca
import numpy as np
import matplotlib.pyplot as plt

genstep = {'specs':False,
		   'remask':False,
		   'autopixplot':False,
		   'pixplot':False,
		   'pixfit':False,
		   'ransig':False,
		   'weight':False}

rcsample = Sample('red_clump',seed=30,order=2,label = 'FE_H',up = -0.4,low=-0.5,fontsize = 10)
rcsample.snrCut()
rcsample.bitmaskData(maskbits = badcombpixmask+2**15)
rcsample.snrCorrect()
rcsample.allPixResiduals()
#rcsample.setPixPlot()
rcsample.allRandomSigma()
rcsample.saveFiles()

'''
weighted = np.zeros((len(elems),len(rcsample.specs)))
weightedsigs = np.zeros((len(elems),len(rcsample.specs)))
i=0
for elem in elems:
	weightedr = rcsample.weighting(rcsample.residual,elem,
								   rcsample.outName('pkl','resids',elem=elem,order = rcsample.order,cross=rcsample.cross))
	weighteds = rcsample.weighting(rcsample.sigma,elem,
								   rcsample.outName('pkl','sigma',elem=elem,order = rcsample.order,seed = rcsample.seed))
	doubleResidualHistPlot(elem,weightedr,weighteds,
						   rcsample.outName('res','residhist',elem = elem,order = rcsample.order,cross=rcsample.cross,seed = rcsample.seed),
						   bins = 50)
	weighted[i] = weightedr
	weightedsigs[i] = weighteds
	i+=1

red_clump_pca = empca(weighted.T)#, 1./np.sqrt(weightedsigs))

plt.figure(figsize = (12,10))
for i in range(len(red_clump_pca.eigvec)):
	plt.plot(red_clump_pca.eigvec[i],label = 'eigenvector {0}'.format(i+1))
for i in range(len(elems)):
	plt.text(i,-0.9,elems[i],color='black')
plt.xlim(-1,15)
plt.legend(loc = 'best')
plt.savefig('red_clump/RCeigvecs.png')

plt.figure(1,figsize=(16,14))
allresplot = np.copy(rcsample.residual.T)
allsigplot = np.copy(rcsample.sigma.T)
ratio = allresplot/allsigplot
mask = np.where(rcsample.mask!=0)
allresplot[mask] = np.nan
ratio[mask] = np.nan
plt.imshow(ratio,aspect = 7214./846.,interpolation='nearest')
plt.colorbar()
plt.savefig('red_clump/NormalizedRCResiduals.png')
plt.close()

plt.figure(5,figsize=(16,14))
plt.imshow(allresplot,aspect = 7214./846.,vmin = -0.05,vmax = 0.05,interpolation='nearest')
plt.colorbar()
plt.savefig('red_clump/RCResiduals.png')
plt.close()

plt.figure(6,figsize=(16,14))
allsigplot[mask] = np.nan
plt.imshow(allsigplot,aspect = 7214./846.,interpolation='nearest')
plt.colorbar()
plt.savefig('red_clump/RCfluxerr.png')
plt.close()

plt.figure(2,figsize=(16,14))
allspecsplot = np.copy(rcsample.specs)
plt.imshow(allspecsplot,aspect = 7214./846.,interpolation='nearest',cmap = 'viridis')
plt.colorbar()
plt.savefig('red_clump/RCspectra.png')
plt.close()

plt.figure(3,figsize=(16,14))
allmasksplot = np.copy(rcsample.mask.astype(np.float64))
allmasksplot[np.where(rcsample.mask==0)] = np.nan
plt.imshow(allmasksplot,aspect = 7214./846.,interpolation='nearest',cmap = 'viridis')
plt.colorbar()
plt.savefig('red_clump/RCmask.png')
plt.close()

plt.figure(4,figsize=(16,14))
allbitmasksplot = np.copy(rcsample.bitmask).astype(np.float64)
allbitmasksplot[np.where(rcsample.bitmask==0)] = np.nan
plt.imshow(np.log2(allbitmasksplot),aspect = 7214./846.,interpolation='nearest',cmap = 'viridis')
plt.colorbar()
plt.savefig('red_clump/RCbitmask.png')
plt.close()
'''

