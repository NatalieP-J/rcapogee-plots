"""

Usage:
run_residuals [-hvpgxS] [-i LABEL] [-u UPLIM] [-d LOWLIM] [-s SAMPTYPE] [-c CLUSTLIS] [-o ORDER]

Sample call:

run_residuals -glx -i FE_H -u -0.1 -d -0.105 -s red_clump -c False

Options:
    -h, --help
    -v, --verbose
    -p, --pixplot                       Option to turn on fit plots at each pixel.
    -g, --generate                      Option to run first sequence (generate everything from scratch)
    -x, --cross                         Option to include cross terms in the fit.
    -S, --save                          Option to save intermediate steps in residual calculation
    -i LABEL, --indep LABEL             A string with a label in which to crop the starsample 
                                        [default: 0]
    -u UPLIM, --upper UPLIM             An upper limit for the starsample crop 
                                        [default: 0]
    -d LOWLIM, --lower LOWLIM           A lower limit for the starsample crop 
                                        [default: 0]
    -s SAMPTYPE, --samptype SAMPTYPE    The type of starsample to run 
                                        [default: clusters]
    -c CLUSTLIS, --clusters CLUSTLIS    A list of subgroupings identified by a key given as the first element in the list
                                        [default: CLUSTER,M67,N2158,N6791,N6819,M13]
    -o ORDER, --order ORDER             Order of polynomial fitting
                                        [default: 2]

"""

# WAVEREGION PLOTS OF EMPCA EIGVECS
import residuals
reload(residuals)
from residuals import Sample,badcombpixmask,aspcappix,tophats,windowPeaks,doubleResidualHistPlot,elems
import os
import time
import numpy as np
from apogee.tools import bitmask
import matplotlib.pyplot as plt
import polyfit as pf
import docopt
from empca import empca
import access_spectrum as acs

################################################################################

# Code used to test residuals.py and confirm all functions are in working order.

################################################################################

def timeIt(fn,*args,**kwargs):
    """
    A basic function to time how long another function takes to run.

    fn:     Function to time.

    Returns the output of the function and its runtime.

    """
    start = time.time()
    output = fn(*args,**kwargs)
    end = time.time()
    return output,end-start

if __name__ == '__main__':
    # Read in command line arguments
    arguments = docopt.docopt(__doc__)

    # Optional Boolean arguments
    verbose = arguments['--verbose']
    pixplot = arguments['--pixplot']
    generate = arguments['--generate']
    crossterm = arguments['--cross']
    savestep = arguments['--save']

    # Optional keyword arguments - convert to appropriate format
    label = arguments['--indep']
    if label == '0':
        label = 0
    up = float(arguments['--upper'])
    low = float(arguments['--lower'])
    order = int(arguments['--order'])
    samptype = arguments['--samptype']
    subgroup_info = arguments['--clusters']
    if subgroup_info == 'False':
        subgroup_info = [False,False]
    elif subgroup_info != 'False':
        subgroup_info = subgroup_info.split(',')
        if len(subgroup_info) == 1:
            if verbose:
                print 'Minimum two elements required: a key to identify the type of subgroup, and a possible entry for the subgroup.'
            warn('Defaulting subgrouping to False')
            subgroup_info = False
    
    if generate:

        # Remove possible cached data.
        if label != 0:
            os.system('rm ./{0}/pickles/*{1}*{2}*{3}*'.format(samptype,label,up,low))
        elif label == 0:
            os.system('rm ./{0}/pickles/*'.format(samptype))

    # Initialize the starsample
    starsample,runtime = timeIt(Sample,samptype,savestep=savestep,order=order,cross=crossterm,label=label,up=up,low=low,subgroup_type=subgroup_info[0],subgroup_lis=subgroup_info[1:],fontsize=10)

    if label != 0:
        statfilename = './{0}/run-statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(starsample.type, starsample.order,starsample.seed,starsample.cross,label,up,low)
    elif label == 0:
         statfilename = './{0}/run-statfile_order{1}_seed{2}_cross{3}.txt'.format(starsample.type, starsample.order,starsample.seed,starsample.cross)
    statfile = open(statfilename,'w+')


    if verbose:
        print '\nInitialization runtime {0:.2f} s'.format(runtime)
        print 'Number of stars {0}\n'.format(starsample.numstars)
    
    if generate:
        statfile.write('################################################################################\n')
        statfile.write('Generate Residuals from Scratch\n')
        statfile.write('################################################################################\n\n')

    if not generate:
        statfile.write('################################################################################\n')
        statfile.write('Load Residuals from File if Possible\n')
        statfile.write('################################################################################\n\n')

    statfile.write('Initialization runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Number of stars {0}\n\n'.format(starsample.numstars))

    # Correct high SNR
    if verbose:
        print 'Maximum SNR before correction {0:.2f}'.format(np.max(starsample.specs/starsample.errs))
    statfile.write('Maximum SNR before correction {0:.2f}\n'.format(np.max(starsample.specs/starsample.errs)))
    noneholder,runtime = timeIt(starsample.snrCorrect)
    if verbose:
        print 'SNR correction runtime {0:.2f} s'.format(runtime)
        print 'Maximum SNR before correction {0:.2f}\n'.format(np.max(starsample.specs/starsample.errs))
    statfile.write('SNR correction runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Maximum SNR before correction {0:.2f}\n\n'.format(np.max(starsample.specs/starsample.errs)))

    # Mask low SNR
    SNRtemp = starsample.specs/starsample.errs
    if verbose:
        print 'Nonzero Minimum SNR before mask {0:.4f}'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)]))
    statfile.write('Nonzero Minimum SNR before mask {0:.4f}\n'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)])))
    noneholder,runtime = timeIt(starsample.snrCut)
    if verbose:
        print 'SNR cut runtime {0:.2f} s'.format(runtime)
        print 'Minimum SNR after mask {0:.4f}\n'.format(np.min(starsample.specs/starsample.errs))
    statfile.write('SNR cut runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Minimum SNR after mask {0:.4f}\n\n'.format(np.min(starsample.specs/starsample.errs)))

    # Apply bitmask
    maskbits = bitmask.bits_set(badcombpixmask+2**15)
    noneholder,runtime = timeIt(starsample.bitmaskData,maskbits = badcombpixmask+2**15)
    if verbose:
        print 'Bitmask application runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Bitmask application runtime {0:.2f} s\n\n'.format(runtime))

    if savestep:
        # Get independent variable arrays
        noneholder,runtime = timeIt(starsample.allIndepVars)
        if verbose:
            print 'Independent variable array generation runtime {0:.2f} s\n'.format(runtime)
        statfile.write('Independent variable array generation runtime {0:.2f} s\n\n'.format(runtime))

        # Do pixel fitting
        noneholder,runtime = timeIt(starsample.allPixFit)
        if verbose:
            print 'Pixel fitting runtime {0:.2f} s\n'.format(runtime)
        statfile.write('Pixel fitting runtime {0:.2f} s\n\n'.format(runtime))

    # Do residual calculation
    noneholder,runtime = timeIt(starsample.allPixResiduals)
    if verbose:
        print 'Pixel residuals runtime {0:.2f} s'.format(runtime)
    statfile.write('Pixel residuals runtime {0:.2f} s\n'.format(runtime))
    if not isinstance(starsample.residual,dict):
        if verbose:
            print 'Maximum residual {0} \n'.format(np.max(starsample.residual))
        statfile.write('Maximum residual {0} \n\n'.format(np.max(starsample.residual)))
    elif isinstance(starsample.residual,dict):
        maxes = []
        for arr in starsample.residual.values():
            maxes.append(np.max(arr[arr.mask==False]))
        if verbose:
            print 'Maximum residual {0} \n'.format(np.max(maxes))
        statfile.write('Maximum residual {0} \n\n'.format(np.max(maxes)))

    # Make plots
    if pixplot:
        noneholder,runtime = timeIt(starsample.setPixPlot)
        if verbose:
            print 'Plotting residuals at window peaks runtime {0:.2f} s\n'.format(runtime)
        statfile.write('Plotting residuals at window peaks runtime {0:.2f} s\n\n'.format(runtime))

    # Gather random sigma
    noneholder,runtime = timeIt(starsample.allRandomSigma)
    if verbose:
        print 'Finding random sigma runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Finding random sigma runtime {0:.2f} s\n\n'.format(runtime))

    starsample.saveFiles()
