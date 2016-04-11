import numpy as np
import matplotlib.pyplot as plt

def R2noises(models,cov,labels,nvecs = 5,pixrange=(5.,50.),deltR2=2e-3,step=1.,numpixs=None,usefile=False):
    if not usefile:
        if not isinstance(numpixs,(list,np.ndarray)):
            numpixs = np.arange(pixrange[0],pixrange[1]+step,step)
        print numpixs
        R2noises = np.zeros((len(numpixs),len(models)+1))
        for n in range(len(numpixs)):
            print 'NUMPIX = ',numpixs[n]
            R2noises[n][0] = numpixs[n]
            corr = models[0].findCorrection(cov,numpix=numpixs[n])
            for m in range(len(models)):
                models[m].pixelEMPCA(nvecs=nvecs,deltR2=deltR2,correction=corr)
                R2noises[n][m+1] = models[m].empcaModelWeight.R2noise
        np.savetxt('numpix_vs_R2.txt',R2noises)
    if usefile:
        R2noises = np.loadtxt('numpix_vs_R2.txt')
    plt.figure()
    plt.axhline(0,color='k',lw=3)
    for m in range(len(models)):
        plt.plot(R2noises[:,0],R2noises[:,m+1],'o-',lw=3,label=labels[m])
    plt.xlabel('number of pixels in smoothing')
    plt.ylabel('R2noise')
    plt.legend(loc='best')
