import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from data import elems,normwindows,elemwindows
import access_spectrum as acs
from residuals_2 import smoothMedian

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)
plt.ion()

def comp_R2(ms,direc=None,labels = ['raw sample,','corrected \nsample,','M.A.D.,']):
    """
    Plot comparison of 
    """
    pres = ['raw sample,','corrected \nsample,','M.A.D.,']
    plt.figure(figsize = (8,8))
    colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, len(ms)))
    styles = [':','-.','--']
    types = ['o','s','^']
    for i in range(len(ms)):
        r2noise_label = '\n$R^2_{\mathrm{noise}}$ = '
        if isinstance(ms[i],str):
            m = acs.pklread(direc+'/'+ms[i])
        else:
            m = ms[i]
        #plt.subplot2grid((1,len(ms)),(0,i))
        plt.ylim(0,1)
        plt.xlim(-1,len(m.R2Array))
        crossvec = np.where(m.R2Array > m.R2noise)
        if crossvec[0] != []:
            crossvec = crossvec[0][0]-1
            plt.axvline(crossvec,color=colors[i],linestyle='-',lw=2)
        plt.plot(m.R2Array,color=colors[i],ls='-',lw=2,ms=7,marker = types[i],label = r'{0} {1} {2:.2f}'.format(pres[i],r2noise_label,m.R2noise))
        #plt.text(1,0.8,r'{0} {1:.2f}'.format(r2noise_label,m.R2noise),fontsize=18)

    plt.ylabel(r'$R^2$',fontsize=22)
    plt.xlabel('number of eigenvectors')
    plt.legend(loc=(0.4,0.5),fontsize=18,frameon=False)
    plt.subplots_adjust(wspace=0, hspace=0)

def show_sample_coverage(models):
    plt.figure(figsize=(10,5.5))
    ax1 = plt.subplot(122,projection='polar')
    ax2 = plt.subplot(121)
    for m in range(len(models)):
        phi = models[m].data['RC_GALPHI']
        r = models[m].data['RC_GALR']
        z = models[m].data['RC_GALZ']
        ax1.plot(phi,r,'ko',markersize=2,alpha=0.2)
        ax1.set_theta_direction(-1)
        ax2.plot(r,z,'ko',markersize=2,alpha=0.2)

    ax1.set_rlabel_position(135)
    ax1.set_rlim(min(r),max(r))
    ax1.set_xticks([])
    ax2.set_xlim(min(r),max(r))
    ax2.set_ylim(min(z),max(z))
    ax2.set_xlabel('R (kpc)')
    ax2.set_ylabel('z (kpc)')
    plt.subplots_adjust(wspace=0.05)

def plot_big_eig(es):
    for e in es:
        med = smoothMedian(e,numpix=100.)
        plt.figure(figsize=(16,5*15))
        for i in range(len(elems)):
            plt.subplot2grid((15,1),(i,0))
            plt.plot(e)
            plt.plot(med,lw=3,color='k')
            plt.xlim(0,7214)
            plt.yticks([])
            plt.plot(0.1*(normwindows[i]/np.max(normwindows[i]))-0.06,color='r')
            if i!=len(elems):
                plt.xticks([])

def plot_comb_eig(es):
    for e in es:
        plt.figure(figsize=(16,5))
        plt.plot(e)
        combwin =np.ma.masked_array(0.1*(np.sum(normwindows,axis=0)/np.max(np.sum(normwindows,axis=0)))-0.06,mask=np.zeros(7214).astype(bool))
        plt.plot(combwin,color='r')
        plt.plot(smoothMedian(e,numpix=100.),lw=3,color='k')
        plt.xlim(0,7214)
            
def plot_eigenvector(es,**kwargs):
    plt.figure(figsize=(14,5))
    plt.axhline(0,color='grey',lw=2)
    plt.xticks(np.arange(len(elems)),elems)
    plt.xlim(-1,len(elems))
    colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, len(es)))
    markers = ['v','d','p']
    i=0
    plt.ylabel('vector magnitude')
    for e in es:
        plt.plot(np.arange(len(elems)),e,'o-',lw=3,color=colors[i],label='eigenvector {0}'.format(i+1),marker=markers[i],markersize=8)
        i+=1
    plt.legend(loc='best',frameon=False)

def plot_example_fit(model,pixel,**kwargs):
    plt.figure(figsize=(8,6))#figsize=(12,6))
    plt.subplot2grid((3,1),(0,0),rowspan=2)
    indeps = model.makeMatrix(pixel)
    fitresult = np.dot(indeps,model.fitCoeffs[pixel].T)
    i = np.array(np.reshape(indeps[:,1],len(fitresult)))[0]
    s = i.argsort()
    plt.plot(i[s],fitresult[s],lw=3,color='k',label='$f(s,T_{\mathrm{eff}}$)')
    plt.errorbar(i[s],model.spectra[:,pixel][s],color='r',fmt='o',yerr=model.spectra_errs[:,pixel][s])
    plt.ylabel('stellar flux $F_p(s)$',fontsize=22)
    plt.xticks([])
    plt.ylim(0.6,1.1)
    plt.yticks(np.arange(0.7,1.1,0.1),np.arange(0.7,1.1,0.1).astype(str))
    plt.legend(loc='best',frameon=False)
    plt.subplot2grid((3,1),(2,0))
    plt.axhline(0,lw=3,color='k')
    plt.errorbar(i[s],model.residuals[:,pixel][s],color='r',fmt='o',yerr=model.spectra_errs[:,pixel][s])
    plt.ylabel('residuals $\delta_p(s)$ ',fontsize=22)
    plt.xlabel(r'$T_{\mathrm{eff}}$ - median($T_{\mathrm{eff}}$) [K]',fontsize=22)
    plt.ylim(-0.05,0.05)
    plt.yticks(np.arange(-0.04,0.05,0.04),np.arange(-0.04,0.05,0.04).astype(str))
    plt.xticks(np.arange(-600,400,200),np.arange(-600,400,200).astype(str))
    plt.subplots_adjust(hspace=0)


def plot_abun(model,elemlist,**kwargs):
    plt.figure(figsize=(9,8))
    i = 0
    for elem in elemlist:
        plt.subplot2grid((len(elemlist),1),(i,0))
        plt.text(4120,0.25,elem)
        ind = elems.index(elem)
        errs = model.abundance_errs[:,ind]
        good = np.where((errs != 0))# & (model.teff[:,0] < 4860.))
        errs = errs[good]
        indeps = np.ones((len(model.teff[:,0][good]),2))
        indeps[:,1] = model.teff[:,0][good]
        elemarr = (model.abundances[:,ind]/np.ma.median(model.abundances[:,ind]))-1
        elemarr = elemarr[good]#/len(np.where(normwindows[ind]!=0)[0])
        Cinv = np.diag(1./errs**2)
        coeffs = np.dot(np.linalg.inv(np.dot(indeps.T,np.dot(Cinv,indeps))),np.dot(indeps.T,np.dot(Cinv,elemarr.T)))
        #coeffs = np.dot(np.linalg.inv(np.dot(indeps.T,indeps)),np.dot(indeps.T,elemarr.T))
        print coeffs
        plt.plot(model.teff[:,0][good],np.dot(indeps,coeffs),'k',lw=3)
        #plt.ylim(-0.4,0.4)
        plt.xlim(3900,5100)
        plt.errorbar(model.teff[:,0][good],elemarr,color='r',yerr=errs,**kwargs)
        plt.yticks([-0.4,-0.2,0,0.2,0.4],['-0.4','-0.2','0','0.2','0.4'])
        plt.text(4120,0.25,'{0}, m={1:.5f}'.format(elem,coeffs[1]))
        plt.xticks([4000,4500,5000],['4000','4500','5000'])
        if i == len(elemlist)-1:
            plt.xlabel('temperature (K)')
        plt.ylabel('deviation')
        i+=1
    plt.subplots_adjust(hspace=0.2)
