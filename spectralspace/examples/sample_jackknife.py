from empca_residuals import *

datadir = './data/'
sample_type = 'red_clump'
bmask = 7935
maxsamp = 5
seed = 44

sample = empca_residuals('apogee',sample_type,maskFilter,ask=True,degree=2,datadir=datadir,
                         badcombpixmask = bmask)

sample.samplesplit(subsamples=25,maxsamp=maxsamp,varfuncs=[meanMed],seed=seed)
