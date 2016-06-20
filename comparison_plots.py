import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from data import elems,normwindows,elemwindows,pixel2element
import access_spectrum as acs
from residuals_2 import smoothMedian

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  16
}

matplotlib.rc('font',**font)
plt.ion()

def comp_R2(ms,direc=None,
            labels = ['raw sample,','corrected \nsample,','M.A.D.,']):
    """
    Plot comparison of R^2 arrays for different models.

    ms:       Either list of model objects or list of strings where
              models are stored.
    direc:    Directory to find model files if ms a list of strings
    labels:   Label corresponding to both models
    """
        
    # Start figure
    plt.figure(figsize = (10,8))
    # Find equally spaced colours from the plasma colourmap
    colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, len(ms)))
    # Choose linestypes and markertypes
    types = ['o','s','^']
    
    vecs = np.array([],dtype=int)
    # For each model
    t=0
    for i in range(len(ms)):
        if t >= len(types):
            t = 0
        # Construct latex label
        r2noise_label = '\n$R^2_{\mathrm{noise}}$ = '
        # Find model - if ms[i] a string, read from file, otherwise use entry
        if isinstance(ms[i],str):
            m = acs.pklread(direc+'/'+ms[i])
        else:
            m = ms[i]
        
        # Set consistent plot boundaries
        plt.ylim(0,1)
        plt.xlim(-1,len(m.R2Array))
        
        # Find where R^2 values cross the R^2_noise line
        crossvec = np.where(m.R2Array > m.R2noise)
        if crossvec[0] != []:
            crossvec = crossvec[0][0]-1
            if crossvec < 0:
                crossvec=0
            plt.axvline(crossvec,0,m.R2Array[crossvec],color=colors[i],
                        linestyle='-',lw=3)
            vecs = np.append(vecs,crossvec)
        
        # Plot R^2 as a function of number of eigenvectors
        plt.plot(m.R2Array,color=colors[i],markerfacecolor=colors[i],
                 markeredgecolor='w',markeredgewidth=2,ls='-',lw=3,ms=14,
                 marker = types[t],
                 label = r'{0} {1} {2:.2f}'.format(labels[i],r2noise_label,
                                                   m.R2noise))
        t+=1
    # Label axes and add the legend
    step=10**np.floor(np.log10(len(m.R2Array)))
    #ticklist = np.concatenate((np.arange(0,len(m.R2Array),step,dtype=int),vecs))
    #plt.xticks(ticklist,ticklist.astype(str))
    for vec in vecs:
        plt.text(vec+0.01*len(m.R2Array),0.02,'{0}'.format(vec))
    plt.ylabel(r'$R^2$',fontsize=22)
    plt.xlabel('number of eigenvectors')
    legend = plt.legend(loc='best',fontsize=18)
    legend.get_frame().set_linewidth(0.0)
    

def plot_big_eig(es):
    """
    For each of the given eigenvectors, produce one plot with 15 subplots 
    each of which shows the same eigenvector with different element windows.

    es:   List of eigenvectors to compare.

    """
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
    """
    For each of the given eigenvectors, produce one plot with the eigenvector
    and the combined window of all elements.
    """
    for e in es:
        plt.figure(figsize=(16,5))
        plt.plot(e)
        combwin =np.ma.masked_array(0.1*(np.sum(normwindows,axis=0)/np.max(np.sum(normwindows,axis=0)))-0.06,mask=np.zeros(7214).astype(bool))
        plt.plot(combwin,color='r')
        plt.plot(smoothMedian(e,numpix=100.),lw=3,color='k')
        plt.xlim(0,7214)

def plot_elem_eig(elem,es):
    """
    For each of the given eigenvectors, produce one plot with the eigenvector
    and the combined window of all elements.
    """
    colours = plt.get_cmap('plasma')(np.linspace(0, 0.85, 3))
    for e in es:
        elemind = elems.index(elem)
        plt.figure(figsize=(16,5))
        plt.plot(e,color=colours[2],lw=3,label='eigenvector')
        win = (0.1*(normwindows[elemind]/np.max(normwindows[elemind]))-0.04)
        plt.plot(win,color=colours[1],lw=3,label='{0} window'.format(elem))
        #plt.plot(smoothMedian(e,numpix=100.),lw=3,color='k')
        legend=plt.legend(loc = 'best')
        legend.get_frame().set_linewidth(0.0)
        plt.xlim(0,7214)
            
def plot_eigenvector(es,labels):
    """
    Compares element form of eigenvectors on the same plot.
    """
    plt.figure(figsize=(14,5))
    plt.axhline(0,color='grey',lw=2)
    plt.xticks(np.arange(len(elems)),elems)
    plt.xlim(-1,len(elems))
    elem_es = pixel2element(es)
    colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, len(es)))
    markers = ['v','d','p','h']
    i=0
    plt.ylabel('vector magnitude')
    for e in elem_es:
        m = i
        if m >= len(markers):
            m = 0
        plt.plot(np.arange(len(elems)),e,'o-',lw=3,color=colors[i],label=labels[i],marker=markers[m],markersize=8)
        i+=1
    plt.legend(loc='best',frameon=False)

def plot_structure(es):
    """
    Plots the Fourier transform of the eigenvectors.
    """
    for e in es:
        f_e = np.fft.rfft(e)
        plt.figure(figsize=(14,5))
        plt.plot(f_e)
