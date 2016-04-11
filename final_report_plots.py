import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  16
}

matplotlib.rc('font',**font)
plt.ion()

def xyz2rphiz(galcoords):
    r = np.sqrt(galcoords.x**2+galcoords.y**2)/1000.
    phi = np.arctan2(galcoords.y,galcoords.x)*(180./np.pi)
    z = galcoords.z/1000.
    return r.value,phi.value,z.value

m67coord = SkyCoord('08h51m18.0s','+11d48m00s',frame='icrs',distance=850*u.pc)
m67_gal = m67coord.galactocentric
m67_gal = xyz2rphiz(m67_gal)
n6819coord = SkyCoord('19h41m18.0s','+40d11m12s',frame='icrs',distance=2208*u.pc)
n6819_gal = n6819coord.galactocentric
n6819_gal = xyz2rphiz(n6819_gal)
m13coord = SkyCoord('16h41m41.634s','+36d27m40.75s',frame='icrs',distance=6800*u.pc)
m13_gal = m13coord.galactocentric
m13_gal = xyz2rphiz(m13_gal)

def comp_R2(m1,m2):
    r2noise_label = '$R^2_{noise}$ = '
    plt.figure()
    plt.subplot2grid((1,2),(0,0))
    plt.ylim(0,1)
    plt.xlim(0,len(m1.R2Array)-1)
    plt.axhline(m1.R2noise,color='k',lw=3)
    plt.plot(m1.R2Array,'o-',lw=3)
    plt.ylabel(r'$R^2$',fontsize=22)
    plt.xlabel('number of eigenvectors')
    plt.text(1,0.8,r'{0} {1:.2f}'.format(r2noise_label,m1.R2noise),fontsize=22)
    plt.title('pre-correction')
    plt.subplot2grid((1,2),(0,1))
    plt.ylim(0,1)
    plt.xlim(0,len(m2.R2Array)-1)
    plt.axhline(m2.R2noise,color='k',lw=3)
    plt.plot(m2.R2Array,'o-',lw=3)
    #plt.ylabel(r'$R^2$',fontsize=22)
    plt.xlabel('number of eigenvectors')
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
        ax2.plot(r,z,'ko',markersize=2,alpha=0.2)

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

    ax1.set_rlabel_position(135)
    ax1.set_rlim(min(r),max(r))
    ax1.set_xticks([])
    ax2.set_xlim(min(r),max(r))
    ax2.set_ylim(min(z),max(z))
    ax2.set_xlabel('R (kpc)')
    ax2.set_ylabel('z (kpc)')
    plt.subplots_adjust(wspace=0.05)

