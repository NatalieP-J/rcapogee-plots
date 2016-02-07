"""

Usage:
midterm_update_plots [-hvx] [-m FNAME] [-s SPECS] [-p PIX] [-a ALPHA]

Options:
    -h, --help
    -v, --verbose
    -x, --hidefigs                  Option to hide figures.
    -s SPECS, --spectra SPECS       Index of spectrum (or list of spectra) to plot 
                                    [default: 0]
    -a ALPHA, --alpha ALPHA         Alpha value for dense plots
                                    [default: 0.5]
    -p PIX, --pixels PIX            Pixels to plot
                                    [default: 0]
    -m FNAME, --model FNAME         Provide a pickle file containing a Sample model (see residuals.py) 
                                    [default: clusters/pickles/model.pkl]

"""

import docopt
import matplotlib
import numpy as np 
import matplotlib.pyplot as plt
import apogee.spec.plot as aplt
import access_spectrum as acs
import polyfit as pf
from residuals import windowPixels
from lowess import lowess

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

aspcappix = 7214

# Stellar properties to fit for different sample types
fitvars = {'clusters':['TEFF'],                    # Fit clusters in effective temperature
           'OCs':['TEFF'],                        # Fit open clusters in effective temperature
           'GCs':['TEFF'],                        # Fit globular clusters in effective temperature
           'red_clump':['TEFF','LOGG','[FE/H]']    # Fit red clump stars in effective temperature, surface gravity and iron abundance
           }


def plot_spec(model,plot,figsize=16.):
    if isinstance(plot,(int)):
        plot = [plot]
    if isinstance(plot,str):
        if plot == 'all':
            plot = np.arange(len(model.data))
        if plot in model.subgroups:
            plot = np.where(model.data[model.subgroup]==plot)[0]
    if isinstance(plot,(list,np.ndarray)):
        if isinstance(plot[0],str):
            newplot = np.array([])
            for string in plot:
                if string in model.subgroups:
                    match = np.where(model.data[model.subgroup]==string)[0]
                    newplot = np.concatenate((newplot,match))
            plot = newplot
        for i in plot:
            if model.subgroup != False:
                subgroup = model.data[model.subgroup][i]
                match = np.where(model.data[model.subgroup]==subgroup)
                ind = i - match[0][0] 
            elif not model.subgroup:
                subgroup = False
                ind = i
            savename = model.outName('plt',content = 'spectrum_{0}'.format(ind),subgroup=subgroup)
            pltspec = model.specs[i]
            pltspec[pltspec.mask] = np.nan
            aplt.waveregions(model.data['LOCATION_ID'][i], model.data['APOGEE_ID'][i],labelLines=False,fig_width=figsize,fig_height=figsize*0.3,label='Unmasked Spectrum')
            aplt.waveregions(pltspec,labelLines=False,overplot=True,label='Masked Spectrum')
            plt.title('Spectra pre and post masking')
            plt.legend(loc='best')
            plt.savefig(savename)
            plt.close()

def plot_res(model,plot,figsize=16.,alpha=0.5,elem=False,fitdata=False,mockdata=False,median=True):
    if isinstance(plot,(int)):
        plot = [plot]
    if isinstance(plot,str):
        if plot == 'all':
            plot = np.arange(len(model.data))
        if plot in model.subgroups:
            plot = np.where(model.data[model.subgroup]==plot)[0]
    if isinstance(plot,(list,np.ndarray)):
        if isinstance(plot[0],str):
            newplot = np.array([])
            for string in plot:
                if string in model.subgroups:
                    match = np.where(model.data[model.subgroup]==string)[0]
                    newplot = np.concatenate((newplot,match))
            plot = newplot
        for i in plot:
            if model.subgroup != False:
                subgroup = model.data[model.subgroup][i]
                match = np.where(model.data[model.subgroup]==subgroup)
                ind = i - match[0][0] 
            elif not model.subgroup:
                subgroup = False
                match = np.where(model.data)
                ind = i
            fitParam,colcode,indeps = model.pixFit(i,match,indeps=False)
            poly = pf.poly(fitParam,colcode,indeps,order = model.order)
            pltres = model.allresid[:,i]
            pltres[pltres.mask] = np.nan
            plt.figure(figsize=(figsize,figsize))
            col = 0
            for indep in indeps:
                plt.subplot2grid((len(indeps),len(indeps)),(0,col),rowspan=2)
                if col == 0:
                    plt.ylabel('Normalized flux')
                sortinds = indep.argsort()
                sortedpoly = poly[sortinds]
                sortedindep = indep[sortinds]
                sortedspec = model.specs[match][:,i][sortinds]
                if fitdata:
                    plt.plot(sortedindep,sortedpoly,'.',color='r',alpha=alpha)
                if median:
                    smoothedfit = np.ma.masked_array(np.zeros(sortedpoly.shape))
                    smoothedfit[sortedindep.mask==False] = lowess(sortedindep[sortedindep.mask == False], sortedpoly[sortedpoly.mask==False], f=2./3., iter=2)
                    plt.plot(sortedindep,smoothedfit,color='r',linewidth=3)
                plt.plot(sortedindep,sortedspec,'.',alpha=alpha,color='b')
                plt.xlim(min(indep),max(indep))
                plt.subplot2grid((len(indeps),len(indeps)),(2,col))
                plt.plot(indep,pltres,'.',alpha=alpha)
                plt.axhline(0,color='k',linewidth=3)
                plt.ylim(-0.1,0.1)
                plt.xlim(min(indep),max(indep))
                plt.ylabel('')
                if col == 0:
                    plt.ylabel('Residual')
                plt.xlabel(fitvars[model.type][col])
                col+=1
            if elem != False:
                savename = model.outName('plt',content = '{0}_residual_{1}'.format(elem,ind),subgroup=subgroup)
                plt.suptitle(elem+' Pixel')
            if not elem:
                plt.suptitle('Unassociated Pixel')
                savename = model.outName('plt',content = 'residual_{0}'.format(ind),subgroup=subgroup)
            plt.savefig(savename)
            plt.close()

if __name__ == '__main__':
    # Read in command line arguments
    arguments = docopt.docopt(__doc__)

    modelname = arguments['--model']

    model=acs.pklread(modelname)

    verbose = arguments['--verbose']
    hide = arguments['--hidefigs']
    if not hide:
        plt.ion()
    elif hide:
        plt.ioff()

    alpha = float(arguments['--alpha'])
    specind = arguments['--spectra']
    try:
        speclist = np.array(specind.split(',')).astype(int)
    except ValueError:
        if specind == 'all':
            speclist = np.arange(model.numstars) # red clump specific

    pix = arguments['--pixels']
    try:
        elem = False
        pixlist = np.array(pix.split(',')).astype(int)
    except ValueError:
        elem = pix
        pixlist = windowPixels[pix][0]

    plot_spec(model,speclist,figsize=12.)
    plot_res(model,pixlist,figsize=12.,alpha=alpha,elem=elem,median=False,fitdata=True)

    


