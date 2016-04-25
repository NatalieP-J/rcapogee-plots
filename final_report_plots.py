import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from data import elems,normwindows
import access_spectrum as acs

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)
plt.ion()

def xyz2rphiz(galcoords):
    r = np.sqrt(galcoords.x**2+galcoords.y**2)/1000.
    phi = np.arctan2(galcoords.y,galcoords.x)*(180./np.pi)
    z = galcoords.z/1000.
    return r.value,phi.value,z.value

"""
m67coord = SkyCoord('08h51m18.0s','+11d48m00s',frame='icrs',distance=850*u.pc)
m67_gal = m67coord.galactocentric
m67_gal = xyz2rphiz(m67_gal)
n6819coord = SkyCoord('19h41m18.0s','+40d11m12s',frame='icrs',distance=2208*u.pc)
n6819_gal = n6819coord.galactocentric
n6819_gal = xyz2rphiz(n6819_gal)
m13coord = SkyCoord('16h41m41.634s','+36d27m40.75s',frame='icrs',distance=6800*u.pc)
m13_gal = m13coord.galactocentric
m13_gal = xyz2rphiz(m13_gal)
"""

def comp_R2(ms,direc=None):
    pres = ['raw sample,','corrected \nsample,','M.A.D.,']
    plt.figure(figsize = (8,5.5))
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
        ax1.plot(-phi,r,'ko',markersize=2,alpha=0.2)
        ax2.plot(r,z,'ko',markersize=2,alpha=0.2)

    """
    ax1.plot(m67_gal[1],m67_gal[0],'ws',markersize=8,mew=2)
    ax2.plot([m67_gal[0],13.6],[m67_gal[2],1.6],'r',lw=2)
    ax2.plot(m67_gal[0],m67_gal[2],'ws',markersize=8,mew=2)
    ax2.text(13.7,1.5,'M67',color='k',fontsize = 12,bbox=dict(facecolor='none', edgecolor='none'))

    ax1.plot(n6819_gal[1],n6819_gal[0],'^w',markersize=8,mew=2)
    ax2.plot([n6819_gal[0],5.7],[n6819_gal[2],-1.3],'r',lw=2)
    ax2.plot(n6819_gal[0],n6819_gal[2],'^w',markersize=8,mew=2)
    ax2.text(2.9,-1.6,'N6819',color='k',fontsize = 12,bbox=dict(facecolor='none', edgecolor='none'))

    ax1.plot(m13_gal[1],m13_gal[0],'pw',markersize=8,mew=2)
    ax2.plot([m13_gal[0],5.7],[m13_gal[2],5.2],'r',lw=2)
    ax2.plot(m13_gal[0],m13_gal[2],'pw',markersize=8,mew=2)
    ax2.text(3.8,5,'M13',color='k',fontsize = 12,bbox=dict(facecolor='none', edgecolor='none'))
    """

    ax1.set_rlabel_position(135)
    ax1.set_rlim(min(r),max(r))
    ax1.set_xticks([])
    ax2.set_xlim(min(r),max(r))
    ax2.set_ylim(min(z),max(z))
    ax2.set_xlabel('R (kpc)')
    ax2.set_ylabel('z (kpc)')
    plt.subplots_adjust(wspace=0.05)

def plot_eigenvector(e,**kwargs):
    #plt.figure(figsize=(14,5))
    plt.xticks(np.arange(len(elems)),elems)
    plt.xlim(-1,len(elems))
    #plt.axhline(0,color='grey',lw=2)
    plt.plot(np.arange(len(elems)),e,'o-',**kwargs)


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

if __name__=='__main__':
    """
    n6819 = fit('clusters',maskFilter,ask=True)
    n6819.findResiduals(gen=True)
    n6819.findAbundances()
    plot_example_fi