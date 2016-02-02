"""

Usage:
midterm_update_plots [-hvx] [-m FNAME] [-s SPECS]

Options:
    -h, --help
    -v, --verbose
    -x, --hidefigs                  Option to hide figures.
    -s SPECS, --spectra SPECS       Index of spectrum (or list of spectra) to plot 
                                    [default: 0]
    -m FNAME, --model FNAME         Provide a pickle file containing a Sample model (see residuals.py) 
                                    [default: clusters/pickles/model.pkl]

"""

import docopt
import numpy as np 
import matplotlib.pyplot as plt
import apogee.spec.plot as aplt
import access_spectrum as acs

aspcappix = 7214

def plot_spec(model,plot):
    if isinstance(plot,(int)):
        plot = [plot]
    if isinstance(plot,'str'):
        if plot == 'all':
            plot = np.arange(len(model.data))
    if isinstance(plot,(list,np.ndarray)):
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
            aplt.waveregions(model.data['LOCATION_ID'], model.data['APOGEE_ID'],labelLines=False,fig_width=16.,fig_height=16.*0.3,overplot=True,label='Unmasked Spectrum')
            aplt.waveregions(pltspec,labelLines=False,fig_width=16.,fig_height=16.*0.3,overplot=True,label='Masked Spectrum')
            plt.title(r'$Spectra\, pre\, and\, post\, masking$')
            plt.legend(loc='best')
            plt.savefig(savename)
            plt.close()

if __name__ == '__main__':
    # Read in command line arguments
    arguments = docopt.docopt(__doc__)

    verbose = arguments['--verbose']
    hide = arguments['--hidefigs']
    if not hide:
        plt.ion()
    elif hide:
        plt.ioff()
    specind = arguments['--spectra']
    speclist = (np.array(specind.split(', ')).astype(int),)

    modelname = arguments['--model']

    model=acs.pklread(modelname)

    


