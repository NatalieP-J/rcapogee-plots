"""

Usage:
run_residuals [-hvpglfmx] [-e NVEC] [-i LABEL] [-u UPLIM] [-d LOWLIM] [-s starSAMPLETYPE] [-c CLUSTLIS] [-o ORDER]

starSample call:

run_residuals -glx -i FE_H -u -0.1 -d -0.105 -s red_clump -c False

Options:
    -h, --help
    -v, --verbose
    -p, --pixplot                       Option to turn on fit plots at each pixel.
    -g, --generate                      Option to run first sequence (generate everything from scratch)
    -l, --loadin                        Option to run second sequence (load from file where possible)
    -x, --cross                         Option to include cross terms in the fit.
    -i LABEL, --indep LABEL             A string with a label in which to crop the starsample 
                                        [default: 0]
    -u UPLIM, --upper UPLIM             An upper limit for the starsample crop 
                                        [default: 0]
    -d LOWLIM, --lower LOWLIM           A lower limit for the starsample crop 
                                        [default: 0]
    -s SAMPTYPE, --samptype SAMPTYPE    The type of starsample to run 
                                        [default: clusters]
    -c CLUSTLIS, --clusters CLUSTLIS    A list of subgroupings identified by a key given as the first element in the list
                                        [default: CLUSTER,M67,N2158,N6791,N6819]
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
    loadin = arguments['--loadin']
    crossterm = arguments['--cross']
    covariance = False #******************

    if pixplot:
        if not generate and not load:
            generate = True 

    if covariance:
        if not generate and not load:
            generate = True

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
    starsample,runtime = timeIt(Sample,samptype,order=order,cross=crossterm,label=label,up=up,low=low,subgroup_type=subgroup_info[0],subgroup_lis=subgroup_info[1:],fontsize=10)

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
        print 'Maximum residual {0} \n'.format(np.max(starsample.residual))
    statfile.write('Pixel residuals runtime {0:.2f} s\n'.format(runtime))
    if not isinstance(starsample.residual,dict):
        statfile.write('Maximum residual {0} \n\n'.format(np.max(starsample.residual)))
    elif isinstance(starsample.residual,dict):
        maxes = []
        for arr in starsample.residual.values():
            maxes.append(np.max(arr[arr.mask==False]))
        statfile.write('Maximum residual {0} \n\n'.format(np.max(maxes)))

        # Calculate covariance matrix
        if covariance:
            # Case with subgroups
            if isinstance(starsample.residual,dict):
                numstars = np.array(starsample.numstars.values())
                # Condition for inclusion in the covariance matrix calculation
                minstar = 9
                condition = numstars > minstar
                # Create output arrays
                # Raw residuals
                allresids = np.ma.masked_array(np.zeros((np.sum(numstars[np.where(condition)]),aspcappix)))
                # Divided by flux uncertainty
                allnormresids = np.ma.masked_array(np.zeros((np.sum(numstars[np.where(condition)]),aspcappix)))
                i = 0
                # Add appropriate stars to output array
                for key in starsample.residual.keys():
                    if starsample.residual[key].shape[1] > minstar:
                        allresids[i:i+starsample.residual[key].shape[1]] = starsample.residual[key].T
                        sigmas = starsample.errs[np.where(starsample.data[starsample.subgroup]==key)]
                        allnormresids[i:i+starsample.residual[key].shape[1]] = starsample.residual[key].T/sigmas
                        i += starsample.residual[key].shape[1]
                # Find covariance
                residcov = np.ma.cov(allresids.T)
                normresidcov = np.ma.cov(allnormresids.T)
                
                # Plot covariance of raw pixel residuals
                plt.figure(figsize=(16,10))
                plt.imshow(residcov,interpolation='nearest',cmap = 'Spectral',vmax=1e-4,vmin=-1e-4)
                plt.colorbar()
                if label != 0:
                    plt.savefig('./{0}/covariance_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross,label,up,low))
                elif label == 0:
                    plt.savefig('./{0}/covariance_order{1}_seed{2}_cross{3}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross))
                plt.close()

                # Plot covariance of residuals divided by pixel flux uncertainty
                plt.figure(figsize=(16,10))
                plt.imshow(normresidcov,interpolation='nearest',cmap = 'Spectral',vmax=4,vmin=-4)
                plt.colorbar()
                if label != 0:
                    plt.savefig('./{0}/normcovariance_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross,label,up,low))
                elif label == 0:
                    plt.savefig('./{0}/normcovariance_order{1}_seed{2}_cross{3}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross))
                plt.close()

                # Plot diagonal of covariance of raw pixel residuals
                plt.figure(figsize=(16,10))
                diag = np.array([residcov[i,i] for i in range(len(residcov))])
                plt.plot(diag)
                plt.xlim(0,len(diag))
                plt.xlabel('Pixel')
                plt.ylabel('Variance')
                if label != 0:
                    plt.savefig('./{0}/covariance_diag_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross,label,up,low))
                elif label == 0:
                    plt.savefig('./{0}/covariance_diag_order{1}_seed{2}_cross{3}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross))
                plt.close()

                # Plot diagonal of  covariance of residuals divided by pixel flux uncertainty
                plt.figure(figsize=(16,10))
                normdiag = np.array([normresidcov[i,i] for i in range(len(normresidcov))])
                plt.plot(diag)
                plt.xlim(0,len(diag))
                plt.xlabel('Pixel')
                plt.ylabel('Variance')
                if label != 0:
                    plt.savefig('./{0}/normcovariance_diag_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross,label,up,low))
                elif label == 0:
                    plt.savefig('./{0}/normcovariance_diag_order{1}_seed{2}_cross{3}.png'.format(starsample.type, starsample.order,starsample.seed,starsample.cross))
                plt.close()

                # Plot two slices of the covariance matrices
                samppix = 3700
                plt.figure(figsize=(16,10))
                plt.plot(residcov[samppix]/np.max(residcov[samppix]),label = 'Raw residual, peak = {0}'.format(np.max(residcov[samppix])))
                plt.plot(normresidcov[samppix]/np.max(normresidcov[samppix]),label = 'Sigma normalized residual, peak = {0}'.format(np.max(normresidcov[samppix])))
                plt.axvline(samppix,color='red')
                plt.ylabel('Covariance at pixel {0} normalized to peak'.format(samppix))
                plt.xlabel('Pixel')
                plt.xlim(samppix-100,samppix+100)
                plt.legend(loc = 'best')
                if label != 0:
                    plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(starsample.type, samppix,starsample.order,starsample.seed,starsample.cross,label,up,low))
                elif label == 0:
                    plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}.png'.format(starsample.type, samppix, starsample.order,starsample.seed,starsample.cross))
                plt.close()
                samppix = 6000
                plt.figure(figsize=(16,10))
                plt.plot(residcov[samppix]/np.max(residcov[samppix]),label = 'Raw residual, peak = {0}'.format(np.max(residcov[samppix])))
                plt.plot(normresidcov[samppix]/np.max(normresidcov[samppix]),label = 'Sigma normalized residual, peak = {0}'.format(np.max(normresidcov[samppix])))
                plt.axvline(samppix,color='red')
                plt.ylabel('Covariance at pixel {0} normalized to peak'.format(samppix))
                plt.xlabel('Pixel')
                plt.xlim(samppix-100,samppix+100)
                plt.legend(loc = 'best')
                if label != 0:
                    plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(starsample.type, samppix,starsample.order,starsample.seed,starsample.cross,label,up,low))
                elif label == 0:
                    plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}.png'.format(starsample.type, samppix, starsample.order,starsample.seed,starsample.cross))
                plt.close()

        # Make plots
        if pixplot:
            noneholder,runtime = timeIt(starsample.setPixPlot)
            if verbose:
                print 'Plotting residuals at window peaks runtime {0:.2f} s\n'.format(runtime)
            statfile.write('Plotting residuals at window peaks runtime {0:.2f} s\n'.format(runtime))

        # Gather random sigma
        noneholder,runtime = timeIt(starsample.allRandomSigma)
        if verbose:
            print 'Finding random sigma runtime {0:.2f} s\n'.format(runtime)
        statfile.write('Finding random sigma runtime {0:.2f} s\n\n'.format(runtime))

        starsample.saveFiles()

        '''
        # Create weighted residuals
        weighted = np.zeros((len(tophats.keys()),len(starsample.specs)))
        weightedsigs = np.zeros((len(tophats.keys()),len(starsample.specs)))
        i=0
        for elem in elems:
            weightedr = starsample.weighting(starsample.rseidual,elem,
                                           starsample.outName('pkl','resids',elem=elem,order = starsample.order,cross=starsample.cross))
            weighteds = starsample.weighting(starsample.sigma,elem,
                                           starsample.outName('pkl','sigma',elem=elem,order = starsample.order,seed = starsample.seed))
            doubleResidualHistPlot(elem,weightedr,weighteds,
                                   starsample.outName('res','residhist',elem = elem,order = starsample.order,cross=starsample.cross,seed = starsample.seed),
                                   bins = 50)
            weighted[i] = weightedr
            weightedsigs[i] = weighteds
            i+=1
        '''
