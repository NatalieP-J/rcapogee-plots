import numpy as np
import matplotlib.pyplot as plt
from apogee.tools import toApStarGrid,pix2wv
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def get_alt_colors(n,cmap='plasma',maxval=0.7):
    """
    Find n alternating colours from a colour map.
    
    n:     Number of colours to get
    cmap:  Colour map to draw colours from.
    
    """
    # Draw colours in order
    colors = plt.get_cmap(cmap)(np.linspace(0,maxval,2))
    # Split colours into first and last half of the colour map
    first = colors[0]
    last = colors[-1]
    colors = np.empty((n,4))
    # Weave colors together
    colors[0::2] = first
    colors[1::2] = last
    return colors

def plot_fullvec(eigvecs,n=5,pixup=8575,pixdown=0,oset=0.08,
                 ncol=5,lw=0.4,**kwargs):
    """
    Plot principal components up to n from model.
    
    eigvecs:   Array of principal components
    n:         Number of PCs to plot
    pixup:     Maximum pixel to plot to
    pixdown:   Minimum pixel to plot to
    oset:      Offset between PCs
    label:     If True, label each PC with its number
    
    Returns a plot of the principal components labelled by rank as a function
    of wavelength between pixup and pixdown.
    """
    # Get alternating line colours
    colors = get_alt_colors(n,**kwargs)
    if n%2 == 0:
        colors=colors[::-1]
    # Tracks where the zeropoint is with respect to true zero
    offset = 0
    # Make ytick label holders
    yticks = np.zeros(n,dtype='S500')
    yticklocs = np.zeros(n)
    # Create subplot
    ax = plt.subplot(111)
    # Convert x-axis to wavelength in microns 
    wvs = pix2wv(np.arange(pixdown,pixup),apStarWavegrid=True)/10**4
    # Plot each principal component as a function of wavelength
    for i in range(n):
        plt.axhline(offset,color='k',lw=1)
        plt.plot(wvs,toApStarGrid(eigvecs[-(i+1)])[pixdown:pixup]+offset,
                 lw=lw,color=colors[i])
        # Track ytick labels and location
        yticks[i] = 'PC {0}'.format(n-i)
        yticklocs[i] = offset
        # Move zeropoint
        offset+=oset
    # Create axis labels
    plt.ylim(-0.003,(n)*oset-0.005)
    plt.xlabel('$\lambda\,\,(\mu\mathrm{m})$',fontsize=23)
    plt.ylabel('normalized flux + constant')
    
    # Tweak ticks
    yminorlocator = MultipleLocator(oset/2)
    ax.yaxis.set_minor_locator(yminorlocator)
    plt.yticks(yticklocs,yticklocs)
    ax.yaxis.set_tick_params(width=2,which='major',size=7)
    ax.yaxis.set_tick_params(width=2,which='minor',size=4)
    plt.twinx()
    scaledyticklocs = ((yticklocs+0.42*oset)/(n*oset-0.002))
    plt.yticks(scaledyticklocs,yticks)
    ax.yaxis.set_tick_params(width=2,which='major',size=7)
    ax.yaxis.set_tick_params(width=2,which='minor',size=4)
    xminorlocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(xminorlocator)
    plt.xticks(fontsize=18)
    xlim=plt.xlim(wvs[0],wvs[-1])
    ax.xaxis.set_tick_params(width=2,which='major',size=7)
    ax.xaxis.set_tick_params(width=2,which='minor',size=4)
