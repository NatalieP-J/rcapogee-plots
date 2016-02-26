"""

Usage:
run_residuals [-hvpgxSL] [-i LABEL] [-u UPLIM] [-d LOWLIM] [-s SAMPTYPE] [-c CLUSTLIS] [-o ORDER] [-C CORRECT] [-l LIMS]

Options:
    -h, --help
    -v, --verbose
    -p, --pixplot                       Option to turn on fit plots at each pixel.
    -g, --generate                      Option to run first sequence (generate everything from scratch)
    -x, --cross                         Option to include cross terms in the fit.
    -S, --save                          Option to save intermediate steps in residual calculation
    -L, --load                          Option to load the model file.
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
    -C CORRECT, --corr CORRECT          File containing correction factor [default: None]
    -l LIMS, --lims LIMS                Signal to noise ratio cutoffs [default: None]

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

# Code used to run residuals.py for an input sample.

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

def weight_residuals(model,residual,sigma,subgroup,numstars):
    weighted = np.ma.masked_array(np.zeros((len(elems),numstars)))
    weightedsigs = np.ma.masked_array(np.zeros((len(elems),numstars)))
    i=0
    for elem in elems:
        weightedr = model.weighting_stars(residual,elem,
                                          model.outName('pkl','resids',elem=elem,
                                                        order = model.order,
                                                        subgroup=subgroup,
                                                        cross=model.cross))
        weighteds = model.weighting_stars(model.sigma.T,elem,
                                          model.outName('pkl','sigma',elem=elem,
                                                        order = model.order,
                                                        subgroup=subgroup,
                                                        seed = model.seed))
        doubleResidualHistPlot(elem,weightedr,weighteds,
                               model.outName('res','residhist',elem = elem,order = model.order,
                                                  cross=model.cross,seed = model.seed,subgroup = subgroup),
                               bins = 50)
        weighted[i] = weightedr
        weightedsigs[i] = weighteds
        i+=1

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
    if subgroup_info != 'None':
        subgroup_info = subgroup_info.split(',')
        if len(subgroup_info) == 1:
            if verbose:
                print 'Minimum two elements required: a key to identify the type of subgroup, and a possible entry for the subgroup.'
            print 'Defaulting subgrouping to False'
            subgroup_info = False
    elif subgroup_info == 'None':
        subgroup_info = [False,False]
    
    correction = arguments['--corr']
    if correction != 'None':
        try:
            correction = float(correction)
            correct='_SNRcorrected_{0}'.format(correction)
        except (TypeError,ValueError):
            correct_name = correction.split('.pkl')[0]
            correct_name = correct_name.split('/')[-1]
            correct='_SNR{0}'.format(correct_name)
            correction = acs.pklread(correction)
    elif correction == 'None':
        correction = None
        correct=False
    
    if generate:
        # Remove possible cached data.
        if label != 0:
            print 'rm {0}/pickles/*{1}*{2}*{3}*.pkl'.format(samptype,label,up,low)
            os.system('rm {0}/pickles/*{1}*{2}*{3}*.pkl'.format(samptype,label,up,low))
        elif label == 0:
            print 'rm {0}/pickles/*.pkl'.format(samptype)
            os.system('rm {0}/pickles/*.pkl'.format(samptype))

    # Initialize the starsample
    starsample,runtime = timeIt(Sample,samptype,savestep=savestep,order=order,cross=crossterm,label=label,up=up,low=low,subgroup_type=subgroup_info[0],subgroup_lis=subgroup_info[1:],fontsize=10,plot=[4],correct=correct)

    if label != 0:
        statfilename = './{0}/run-statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(starsample.type, starsample.order,starsample.seed,starsample.cross,label,up,low)
    elif label == 0:
         statfilename = './{0}/run-statfile_order{1}_seed{2}_cross{3}.txt'.format(starsample.type, starsample.order,starsample.seed,starsample.cross)
    statfile = open(statfilename,'w+')

    if not starsample.subgroup:
        totalstars = starsample.numstars
    elif starsample.subgroup != False:
        try:
            totalstars = np.sum(starsample.numstars.values())
        except AttributeError:
            totalstars = starsample.numstars

    if verbose:
        print '\nInitialization runtime {0:.2f} s'.format(runtime)
        print 'Number of stars {0}\n'.format(totalstars)

    assert totalstars > 0.
    
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

    # SNR
    lims = arguments['--lims']
    lims = lims.split(',')
    if len(lims) == 1:
        if lims[0] == 'None':
            lowSNR = np.min(starsample.specs/starsample.errs)
            upSNR = np.max(starsample.specs/starsample.errs)
        elif lims[0] != 'None':
            lowSNR = float(lims[0])
            upSNR = np.max(starsample.specs/starsample.errs)
    elif len(lims) > 1:
        if len(lims) != 2:
            if verbose:
                print 'Too many SNR cuts specified, only first two used'
            statfile.write('Too many SNR cuts specified, only first two used')
        if lims[0] == 'None':
            lowSNR = np.min(starsample.specs/starsample.errs)
            if lims[1] == 'None':
                upSNR = np.max(starsample.specs/starsample.errs)
            elif lims[1] != 'None':
                upSNR = float(lims[1])
        elif lims[0] != 'None':
            lowSNR = float(lims[0])
            if lims[1] == 'None':
                upSNR = np.max(starsample.specs/starsample.errs)
            elif lims[1] != 'None':
                upSNR = float(lims[1])


    # Mask SNR
    SNRtemp = starsample.specs/starsample.errs
    if verbose:
        print 'Nonzero Minimum SNR before mask {0:.4f}'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)]))
        print 'Maximum SNR before correction {0:.2f}'.format(np.max(starsample.specs/starsample.errs))
    statfile.write('Maximum SNR before correction {0:.2f}\n'.format(np.max(starsample.specs/starsample.errs)))
    statfile.write('Nonzero Minimum SNR before mask {0:.4f}\n'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)])))
    
    noneholder,runtime = timeIt(starsample.snrCut,low=lowSNR,up=upSNR)
    if verbose:
        print 'SNR cut runtime {0:.2f} s'.format(runtime)
        print 'Minimum SNR after mask {0:.4f}'.format(np.min(starsample.specs/starsample.errs))
        print 'Maximum SNR after correction {0:.2f}\n'.format(np.max(starsample.specs/starsample.errs))
    statfile.write('SNR cut runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Minimum SNR after mask {0:.4f}\n'.format(np.min(starsample.specs/starsample.errs)))
    statfile.write('Maximum SNR after correction {0:.2f}\n\n'.format(np.max(starsample.specs/starsample.errs)))

    # Correct SNR if necessary
    noneholder,runtime = timeIt(starsample.snrCorrect,corr_fact=correction)
    if verbose:
        print 'SNR correction runtime {0:.2f} s\n'.format(runtime)
    statfile.write('SNR correction runtime {0:.2f} s\n'.format(runtime))

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
    
    statfile.close()

    if not starsample.subgroup:
        starsample.allresid = starsample.residual.T

    elif starsample.subgroup != False:
        totalstars = np.sum(starsample.numstars.values())
        starsample.allresid = np.ma.masked_array(np.zeros((totalstars,aspcappix)))
        j = 0
        for subgroup in starsample.subgroups:
            starsample.allresid[j:j+starsample.numstars[subgroup]] = starsample.residual[subgroup].T
            j+=starsample.numstars[subgroup]

    starsample.modelname = starsample.outName('pkl',content = '/models/model')
    acs.pklwrite(starsample.modelname,starsample)


