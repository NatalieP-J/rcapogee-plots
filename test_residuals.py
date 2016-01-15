"""

Usage:
test_residuals [-hvpglfmx] [-e NVEC] [-i LABEL] [-u UPLIM] [-d LOWLIM] [-s SAMPLETYPE] [-c CLUSTLIS] [-o ORDER]

Sample call:

test_residuals -gelx -i FE_H -u -0.1 -d -0.105 -s red_clump -c False

Options:
    -h, --help
    -v, --verbose
    -p, --pixplot                       Option to turn on fit plots at each pixel.
    -g, --generate                      Option to run first sequence (generate everything from scratch)
    -l, --loadin                        Option to run second sequence (load from file where possible)
    -f, --fittest                       Option to run third sequence (test polynomial fits)
    -m, --matrix                        Option to create covariance matrix
    -x, --cross                         Option to include cross terms in the fit.
    -e NVEC, --empca NVEC               An integer specifying how many vectors to use in empca. 
                                       [default: False]
    -i LABEL, --indep LABEL             A string with a label in which to crop the sample 
                                        [default: 0]
    -u UPLIM, --upper UPLIM             An upper limit for the sample crop 
                                        [default: 0]
    -d LOWLIM, --lower LOWLIM           A lower limit for the sample crop 
                                        [default: 0]
    -s SAMPTYPE, --samptype SAMPTYPE    The type of sample to run 
                                        [default: clusters]
    -c CLUSTLIS, --clusters CLUSTLIS    A list of subgroupings identified by a key given as the first element in the list
                                        [default: CLUSTER,M67,N2158,N6791,N6819]
    -o ORDER, --order ORDER             Order of polynomial fitting
                                        [default: 2]

"""

# WAVEREGION PLOTS OF EMPCA EIGVECS

from residuals import Sample,badcombpixmask,aspcappix,tophats,windowPeaks,doubleResidualHistPlot,elems
import os
import time
import numpy as np
from apogee.tools import bitmask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polyfit as pf
import docopt
from empca import empca
import access_spectrum as acs

# Read in command line arguments
arguments = docopt.docopt(__doc__)

# Optional Boolean arguments
verbose = arguments['--verbose']
pixplot = arguments['--pixplot']
run1 = arguments['--generate']
run2 = arguments['--loadin']
run3 = arguments['--fittest']
covariance = arguments['--matrix']
cross = arguments['--cross']

empcarun = arguments['--empca']
if empcarun == 'False':
    empcarun = False

elif empcarun != 'False':
    nvec = int(empcarun)
    empcarun = True
    if not run1 and not run2:
        run1 = True

if pixplot:
    if not run1 and not run2:
        run1 = True 

if covariance:
    if not run1 and not run2:
        run1 = True

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

if run1:

    # Remove possible cached data.
    if label != 0:
        os.system('rm ./{0}/pickles/*{1}*{2}*{3}*'.format(samptype,label,up,low))
    elif label == 0:
        os.system('rm ./{0}/pickles/*'.format(samptype))

    # Initialize the sample
    testsample,runtime = timeIt(Sample,samptype,order=order,cross=cross,label=label,up=up,low=low,subgroup_type=subgroup_info[0],subgroup_lis=subgroup_info[1:],fontsize=10)

    if label != 0:
        statfilename = './{0}/test-statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low)
    elif label == 0:
        statfilename = './{0}/test-statfile_order{1}_seed{2}_cross{3}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross)
    statfile = open(statfilename,'w+')

    if verbose:
        print '\nInitialization runtime {0:.2f} s'.format(runtime)
        print 'Number of stars {0}\n'.format(testsample.numstars)
    
    try:
        statfile.write('################################################################################\n')
        statfile.write('Run 1 - Time Processes From Scratch\n')
        statfile.write('################################################################################\n\n')
    except NameError:
        statfile = open(statfilename,'w+')
        statfile.write('################################################################################\n')
        statfile.write('Run 1 - Time Processes From Scratch\n')
        statfile.write('################################################################################\n\n')

    statfile.write('Initialization runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Number of stars {0}\n\n'.format(testsample.numstars))

    # Correct high SNR
    if verbose:
        print 'Maximum SNR before correction {0:.2f}'.format(np.max(testsample.specs/testsample.errs))
    statfile.write('Maximum SNR before correction {0:.2f}\n'.format(np.max(testsample.specs/testsample.errs)))
    noneholder,runtime = timeIt(testsample.snrCorrect)
    if verbose:
        print 'SNR correction runtime {0:.2f} s'.format(runtime)
        print 'Maximum SNR before correction {0:.2f}\n'.format(np.max(testsample.specs/testsample.errs))
    statfile.write('SNR correction runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Maximum SNR before correction {0:.2f}\n\n'.format(np.max(testsample.specs/testsample.errs)))

    # Mask low SNR
    SNRtemp = testsample.specs/testsample.errs
    if verbose:
        print 'Nonzero Minimum SNR before mask {0:.4f}'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)]))
    statfile.write('Nonzero Minimum SNR before mask {0:.4f}\n'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)])))
    noneholder,runtime = timeIt(testsample.snrCut)
    if verbose:
        print 'SNR cut runtime {0:.2f} s'.format(runtime)
        print 'Minimum SNR after mask {0:.4f}\n'.format(np.min(testsample.specs/testsample.errs))
    statfile.write('SNR cut runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Minimum SNR after mask {0:.4f}\n\n'.format(np.min(testsample.specs/testsample.errs)))
    try:
        assert np.array_equal(testsample.specs.mask,testsample.errs.mask)
    except AssertionError:
        if verbose:
            print 'Spectra and uncertainty array do not have the same mask.\n'
        statfile.write('Spectra and uncertainty array do not have the same mask.\n\n')
    try:
        assert np.array_equal(np.where(testsample.specs.data/testsample.errs.data < 50),np.where(testsample.specs.mask == True))
    except AssertionError:
        if verbose:
            print 'SNR masking failed.\n'
        statfile.write('SNR masking failed.\n\n')

    # Apply bitmask
    maskbits = bitmask.bits_set(badcombpixmask+2**15)
    noneholder,runtime = timeIt(testsample.bitmaskData,maskbits = badcombpixmask+2**15)
    if verbose:
        print 'Bitmask application runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Bitmask application runtime {0:.2f} s\n\n'.format(runtime))
    for bit in maskbits:
        inds = bitmask.bit_set(bit,testsample.bitmask)
        failset = np.where(testsample.specs.mask[inds]==False)
        try:
            assert len(failset[0])==0
        except AssertionError:
            if verbose:
                print 'Mask on bit {0} failed\n'.format(bit)
            statfile.write('Mask on bit {0} failed\n\n'.format(bit))
    try:
        assert np.array_equal(testsample.specs.mask,testsample.errs.mask)
    except AssertionError:
        if verbose:
            print 'Spectra and uncertainty array do not have the same mask.\n'
        statfile.write('Spectra and uncertainty array do not have the same mask.\n\n')

    # Get independent variable arrays
    noneholder,runtime = timeIt(testsample.allIndepVars)
    if verbose:
        print 'Independent variable array generation runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Independent variable array generation runtime {0:.2f} s\n\n'.format(runtime))
    # Case with no subgroups
    if not isinstance(testsample.allindeps,dict):
        for i in range(len(testsample.allindeps)):
            for indep in testsample.allindeps[i]:
                try:
                    assert np.array_equal(testsample.specs.mask[:,i],indep.mask)
                except AssertionError:
                    if verbose:
                        print 'Independent variables improperly masked.\n'
                    statfile.write('Independent variables improperly masked at pixel {0} \n\n'.format(i))
    # Case with subgroups
    if isinstance(testsample.allindeps,dict):
        for key in testsample.allindeps.keys():
            match = np.where(testsample.data[testsample.subgroup]==key)
            for i in range(len(testsample.allindeps[key])):
                for indep in testsample.allindeps[key][i]:
                    try:
                        assert np.array_equal(testsample.specs.mask[match][:,i],indep.mask)
                    except AssertionError:
                        if verbose:
                            print 'Independent variables improperly masked.\n'
                        statfile.write('Independent variables improperly masked at pixel {0} \n\n'.format(i))

    # Do pixel fitting
    noneholder,runtime = timeIt(testsample.allPixFit)
    if verbose:
        print 'Pixel fitting runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Pixel fitting runtime {0:.2f} s\n\n'.format(runtime))

    # Do residual calculation
    noneholder,runtime = timeIt(testsample.allPixResiduals)
    if verbose:
        print 'Pixel residuals runtime {0:.2f} s'.format(runtime)
        print 'Maximum residual {0} \n'.format(np.max(testsample.residual))
    statfile.write('Pixel residuals runtime {0:.2f} s\n'.format(runtime))
    if not isinstance(testsample.residual,dict):
        statfile.write('Maximum residual {0} \n\n'.format(np.max(testsample.residual)))
    elif isinstance(testsample.residual,dict):
        maxes = []
        for arr in testsample.residual.values():
            maxes.append(np.max(arr[arr.mask==False]))
        statfile.write('Maximum residual {0} \n\n'.format(np.max(maxes)))

    # Calculate covariance matrix
    if covariance:
        # Case with subgroups
        if isinstance(testsample.residual,dict):
            numstars = np.array(testsample.numstars.values())
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
            for key in testsample.residual.keys():
                if testsample.residual[key].shape[1] > minstar:
                    allresids[i:i+testsample.residual[key].shape[1]] = testsample.residual[key].T
                    sigmas = testsample.errs[np.where(testsample.data[testsample.subgroup]==key)]
                    allnormresids[i:i+testsample.residual[key].shape[1]] = testsample.residual[key].T/sigmas
                    i += testsample.residual[key].shape[1]
            # Find covariance
            residcov = np.ma.cov(allresids.T)
            normresidcov = np.ma.cov(allnormresids.T)
            
            # Plot covariance of raw pixel residuals
            plt.figure(figsize=(16,10))
            plt.imshow(residcov,interpolation='nearest',cmap = 'Spectral',vmax=1e-4,vmin=-1e-4)
            plt.colorbar()
            if label != 0:
                plt.savefig('./{0}/covariance_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariance_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()

            # Plot covariance of residuals divided by pixel flux uncertainty
            plt.figure(figsize=(16,10))
            plt.imshow(normresidcov,interpolation='nearest',cmap = 'Spectral',vmax=4,vmin=-4)
            plt.colorbar()
            if label != 0:
                plt.savefig('./{0}/normcovariance_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/normcovariance_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()

            # Plot diagonal of covariance of raw pixel residuals
            plt.figure(figsize=(16,10))
            diag = np.array([residcov[i,i] for i in range(len(residcov))])
            plt.plot(diag)
            plt.xlim(0,len(diag))
            plt.xlabel('Pixel')
            plt.ylabel('Variance')
            if label != 0:
                plt.savefig('./{0}/covariance_diag_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariance_diag_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()

            # Plot diagonal of  covariance of residuals divided by pixel flux uncertainty
            plt.figure(figsize=(16,10))
            normdiag = np.array([normresidcov[i,i] for i in range(len(normresidcov))])
            plt.plot(diag)
            plt.xlim(0,len(diag))
            plt.xlabel('Pixel')
            plt.ylabel('Variance')
            if label != 0:
                plt.savefig('./{0}/normcovariance_diag_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/normcovariance_diag_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
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
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(testsample.type, samppix,testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}.png'.format(testsample.type, samppix, testsample.order,testsample.seed,testsample.cross))
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
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(testsample.type, samppix,testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}.png'.format(testsample.type, samppix, testsample.order,testsample.seed,testsample.cross))
            plt.close()

    # Make plots
    if pixplot:
        noneholder,runtime = timeIt(testsample.setPixPlot)
        if verbose:
            print 'Plotting residuals at window peaks runtime {0:.2f} s\n'.format(runtime)
        statfile.write('Plotting residuals at window peaks runtime {0:.2f} s\n'.format(runtime))

    # Gather random sigma
    noneholder,runtime = timeIt(testsample.allRandomSigma)
    if verbose:
        print 'Finding random sigma runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Finding random sigma runtime {0:.2f} s\n\n'.format(runtime))

    testsample.saveFiles()

    # Get empca results
    if empcarun:
        if not isinstance(testsample.residual,dict):
            empcaname = testsample.outName('pkl',content = 'empca',order = testsample.order,seed = testsample.seed,cross=testsample.cross)
            empcamodel,runtime = timeIt(empca,testsample.residual.T)
            empcamodel_weight,runtime = timeIt(empca,testsample.residual.T,1./(testsample.errs**2))
            acs.pklwrite(empcaname,[empcamodel,empcamodel_weight])
        elif isinstance(testsample.residual,dict):
            for subgroup in testsample.subgroups:
                empcaname = testsample.outName('pkl',subgroup = subgroup,content = 'empca',order = testsample.order,seed = testsample.seed,cross=testsample.cross)
                match = np.where(testsample.data[testsample.subgroup] == subgroup)
                empcamodel,runtime = timeIt(empca,testsample.residual[subgroup].T)
                empcamodel_weight,runtime = timeIt(empca,testsample.residual[subgroup].T,1./(testsample.errs[match]**2))
                acs.pklwrite(empcaname,[empcamodel,empcamodel_weight])
        if verbose:
            print 'EMPCA runtime {0:.2f} s\n'.format(runtime)
        statfile.write('EMPCA runtime {0:.2f} s\n\n'.format(runtime))

################################################################################

# Plot both Boolean mask and updated bitmask.

################################################################################
    
    if not isinstance(testsample.numstars,dict):
        totalstars = testsample.numstars
        match = np.where(data)
    elif isinstance(testsample.numstars,dict):
        totalstars = np.sum(testsample.numstars.values())
        match = (np.array([],dtype=np.int),)
        for subgroup in testsample.subgroups:
            newind = np.where(testsample.data[testsample.subgroup] == subgroup)
            match = (np.concatenate((match[0],newind[0])),)


    plt.figure(1,figsize=(16,14))
    allmasksplot = np.copy(testsample.mask[match].astype(np.float64))
    allmasksplot[np.where(testsample.mask[match]==0)] = np.nan
    plt.imshow(allmasksplot,aspect = 7214./totalstars,interpolation='nearest',cmap = 'viridis')
    plt.ylim(0,totalstars)
    plt.colorbar()
    if label != 0:
        plt.savefig('./{0}/test1_mask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
    elif label == 0:
        plt.savefig('./{0}/test1_mask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
    plt.close()

    plt.figure(2,figsize=(16,14))
    allbitmasksplot = np.copy(testsample.bitmask[match]).astype(np.float64)
    allbitmasksplot[np.where(testsample.bitmask[match]==0)] = np.nan
    plt.imshow(np.log2(allbitmasksplot),aspect = 7214./totalstars,interpolation='nearest',cmap = 'viridis')
    plt.ylim(0,totalstars)
    plt.colorbar()
    if label != 0:
        plt.savefig('./{0}/test1_bitmask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
    elif label == 0:
        plt.savefig('./{0}/test1_bitmask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
    plt.close()

    plt.figure(3,figsize=(16,14))
    SNRplot = testsample.specs[match]/testsample.errs[match]
    SNRplot[np.where(SNRplot.mask!=0)] = np.nan
    plt.imshow(SNRplot,aspect = 7214./totalstars,interpolation='nearest',cmap = 'viridis')
    plt.ylim(0,totalstars)
    plt.colorbar()
    if label != 0:
        plt.savefig('./{0}/test1_SNR_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
    elif label == 0:
        plt.savefig('./{0}/test1_SNR_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
    plt.close()

################################################################################

# Re-run all steps to check that caching data reduces run time.

################################################################################

if run2:

    if verbose:
        print '\n\nLoading from saved files\n'

    # Initialize the sample
    testsample,runtime = timeIt(Sample,samptype,order=order,cross=cross,label=label,up=up,low=low,subgroup_type=subgroup_info[0],subgroup_lis=subgroup_info[1:],fontsize=10)
    if verbose:
        print '\nInitialization runtime {0:.2f} s'.format(runtime)
        print 'Number of stars {0}\n'.format(testsample.numstars)

    if label != 0:
        statfilename = './{0}/test-statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low)
    elif label == 0:
         statfilename = './{0}/test-statfile_order{1}_seed{2}_cross{3}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross)
    
    try:
        statfile.write('################################################################################\n')
        statfile.write('Run 2 - Time Processes Loading From Files \n')
        statfile.write('################################################################################\n\n')
    except NameError:
        statfile = open(statfilename,'w+')
        statfile.write('################################################################################\n')
        statfile.write('Run 2 - Time Processes Loading From Files \n')
        statfile.write('################################################################################\n\n')

    statfile.write('Initialization runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Number of stars {0}\n\n'.format(testsample.numstars))

    # Correct high SNR
    if verbose:
        print 'Maximum SNR before correction {0:.2f}'.format(np.max(testsample.specs/testsample.errs))
    statfile.write('Maximum SNR before correction {0:.2f}\n'.format(np.max(testsample.specs/testsample.errs)))
    noneholder,runtime = timeIt(testsample.snrCorrect)
    if verbose:
        print 'SNR correction runtime {0:.2f} s'.format(runtime)
        print 'Maximum SNR before correction {0:.2f}\n'.format(np.max(testsample.specs/testsample.errs))
    statfile.write('SNR correction runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Maximum SNR before correction {0:.2f}\n\n'.format(np.max(testsample.specs/testsample.errs)))

    # Mask low SNR
    SNRtemp = testsample.specs/testsample.errs
    if verbose:
        print 'Nonzero Minimum SNR before mask {0:.4f}'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)]))
    statfile.write('Nonzero Minimum SNR before mask {0:.4f}\n'.format(np.min(SNRtemp[np.where(SNRtemp > 1e-5)])))
    noneholder,runtime = timeIt(testsample.snrCut)
    if verbose:
        print 'SNR cut runtime {0:.2f} s'.format(runtime)
        print 'Minimum SNR after mask {0:.4f}\n'.format(np.min(testsample.specs/testsample.errs))
    statfile.write('SNR cut runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Minimum SNR after mask {0:.4f}\n\n'.format(np.min(testsample.specs/testsample.errs)))
    try:
        assert np.array_equal(testsample.specs.mask,testsample.errs.mask)
    except AssertionError:
        if verbose:
            print 'Spectra and uncertainty array do not have the same mask.\n'
        statfile.write('Spectra and uncertainty array do not have the same mask.\n\n')
    try:
        assert all(testsample.specs.mask[np.where(testsample.specs.data/testsample.errs.data < 50)]==True)
    except AssertionError:
        if verbose:
            print 'SNR masking failed.\n'
        statfile.write('SNR masking failed.\n\n')

    # Apply bitmask
    maskbits = bitmask.bits_set(badcombpixmask+2**15)
    noneholder,runtime = timeIt(testsample.bitmaskData,maskbits = badcombpixmask+2**15)
    if verbose:
        print 'Bitmask application runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Bitmask application runtime {0:.2f} s\n\n'.format(runtime))
    for bit in maskbits:
        inds = bitmask.bit_set(bit,testsample.bitmask)
        failset = np.where(testsample.specs.mask[inds]==False)
        try:
            assert len(failset[0])==0
        except AssertionError:
            if verbose:
                print 'Mask on bit {0} failed\n'.format(bit)
            statfile.write('Mask on bit {0} failed\n\n'.format(bit))
    try:
        assert np.array_equal(testsample.specs.mask,testsample.errs.mask)
    except AssertionError:
        if verbose:
            print 'Spectra and uncertainty array do not have the same mask.\n'
        statfile.write('Spectra and uncertainty array do not have the same mask.\n\n')

    # Get independent variable arrays
    noneholder,runtime = timeIt(testsample.allIndepVars)
    if verbose:
        print 'Independent variable array generation runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Independent variable array generation runtime {0:.2f} s\n\n'.format(runtime))

    # Do pixel fitting
    noneholder,runtime = timeIt(testsample.allPixFit)
    if verbose:
        print 'Pixel fitting runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Pixel fitting runtime {0:.2f} s\n\n'.format(runtime))

    # Do residual calculation
    noneholder,runtime = timeIt(testsample.allPixResiduals)
    if verbose:
        print 'Pixel residuals runtime {0:.2f} s'.format(runtime)
        print 'Maximum residual {0} \n'.format(np.max(testsample.residual))
    statfile.write('Pixel residuals runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Maximum residual {0} \n\n'.format(np.max(testsample.residual)))

    # Gather random sigma
    noneholder,runtime = timeIt(testsample.allRandomSigma)
    if verbose:
        print 'Finding random sigma runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Finding random sigma runtime {0:.2f} s\n\n'.format(runtime))

    # Calculate covariance matrix
    if covariance:
        # Case with subgroups
        if isinstance(testsample.residual,dict):
            numstars = np.array(testsample.numstars.values())
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
            for key in testsample.residual.keys():
                if testsample.residual[key].shape[1] > minstar:
                    allresids[i:i+testsample.residual[key].shape[1]] = testsample.residual[key].T
                    sigmas = testsample.errs[np.where(testsample.data[testsample.subgroup]==key)]
                    allnormresids[i:i+testsample.residual[key].shape[1]] = testsample.residual[key].T/sigmas
                    i += testsample.residual[key].shape[1]
            # Find covariance
            residcov = np.ma.cov(allresids.T)
            normresidcov = np.ma.cov(allnormresids.T)
            
            # Plot covariance of raw pixel residuals
            plt.figure(figsize=(16,10))
            plt.imshow(residcov,interpolation='nearest',cmap = 'Spectral',vmax=1e-4,vmin=-1e-4)
            plt.colorbar()
            if label != 0:
                plt.savefig('./{0}/covariance_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariance_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()

            # Plot covariance of residuals divided by pixel flux uncertainty
            plt.figure(figsize=(16,10))
            plt.imshow(normresidcov,interpolation='nearest',cmap = 'Spectral',vmax=4,vmin=-4)
            plt.colorbar()
            if label != 0:
                plt.savefig('./{0}/normcovariance_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/normcovariance_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()

            # Plot diagonal of covariance of raw pixel residuals
            plt.figure(figsize=(16,10))
            diag = np.array([residcov[i,i] for i in range(len(residcov))])
            plt.plot(diag)
            plt.xlim(0,len(diag))
            plt.xlabel('Pixel')
            plt.ylabel('Variance')
            if label != 0:
                plt.savefig('./{0}/covariance_diag_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariance_diag_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()

            # Plot diagonal of  covariance of residuals divided by pixel flux uncertainty
            plt.figure(figsize=(16,10))
            normdiag = np.array([normresidcov[i,i] for i in range(len(normresidcov))])
            plt.plot(normdiag)
            plt.xlim(0,len(normdiag))
            plt.xlabel('Pixel')
            plt.ylabel('Variance')
            if label != 0:
                plt.savefig('./{0}/normcovariance_diag_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/normcovariance_diag_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
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
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(testsample.type, samppix,testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}.png'.format(testsample.type, samppix, testsample.order,testsample.seed,testsample.cross))
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
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(testsample.type, samppix,testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/covariancepix{1}_order{2}_seed{3}_cross{4}.png'.format(testsample.type, samppix, testsample.order,testsample.seed,testsample.cross))
            plt.close()

    '''
    # Create weighted residuals
    weighted = np.zeros((len(tophats.keys()),len(testsample.specs)))
    weightedsigs = np.zeros((len(tophats.keys()),len(testsample.specs)))
    i=0
    for elem in elems:
        weightedr = testsample.weighting(testsample.residual,elem,
                                       testsample.outName('pkl','resids',elem=elem,order = testsample.order,cross=testsample.cross))
        weighteds = testsample.weighting(testsample.sigma,elem,
                                       testsample.outName('pkl','sigma',elem=elem,order = testsample.order,seed = testsample.seed))
        doubleResidualHistPlot(elem,weightedr,weighteds,
                               testsample.outName('res','residhist',elem = elem,order = testsample.order,cross=testsample.cross,seed = testsample.seed),
                               bins = 50)
        weighted[i] = weightedr
        weightedsigs[i] = weighteds
        i+=1
    '''

    # Get empca results
    if empcarun:
        if not isinstance(testsample.residual,dict):
            empcaname = testsample.outName('pkl',content = 'empca',order = testsample.order,cross=testsample.cross)
            if os.path.isfile(empcaname):
                empcamodel_weight = acs.pklread(empcaname)
            elif not os.path.isfile(empcaname):
                #empcamodel,runtime = timeIt(empca,testsample.residual.T,nvec=nvec)
                empcamodel_weight,runtime = timeIt(empca,testsample.residual.T,1./(testsample.errs**2),nvec=nvec)
                totalruntime = runtime
                acs.pklwrite(empcaname,empcamodel_weight)
                if verbose:
                    print 'EMPCA runtime {0:.2f} s\n'.format(totalruntime)
                statfile.write('EMPCA runtime {0:.2f} s\n\n'.format(totalruntime))
            R2noise = 1 - np.var(testsample.errs)/np.var(testsample.residual.T)
            R2 = np.zeros(nvec)
            for vec in range(nvec):
                R2[vec] = empcamodel_weight.R2(vec)

            # Plot calculated R2
            plt.figure(1,figsize=(16,14))
            plt.plot(range(nvec),R2)
            plt.axhline(R2noise,color='red')
            plt.xlabel('Number of vectors')
            plt.ylabel('Fraction of variance accounted for')
            plt.text(0,R2noise+0.01,'Fraction of variance above noise')
            if label != 0:
                plt.savefig('./{0}/empcaR2_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
            elif label == 0:
                plt.savefig('./{0}/empcaR2_mask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
            plt.close()


        elif isinstance(testsample.residual,dict):
            for subgroup in testsample.subgroups:
                empcaname = testsample.outName('pkl',subgroup = subgroup,content = 'empca',order = testsample.order,seed = testsample.seed,cross=testsample.cross)
                totalruntime = 0
                if os.path.isfile(empcaname):
                    empcamodel,empcamodel_weight = acs.pklread(empcaname)
                elif not os.path.isfile(empcaname):
                    match = np.where(testsample.data[testsample.subgroup] == subgroup)
                    empcamodel,runtime = timeIt(empca,testsample.residual[subgroup].T,nvec=nvec)
                    empcamodel_weight,runtime = timeIt(empca,testsample.residual[subgroup].T,1./(testsample.errs[match]**2),nvec=nvec)
                    totalruntime += runtime
                    acs.pklwrite(empcaname,[empcamodel,empcamodel_weight])
                if verbose:
                    print 'EMPCA runtime {0:.2f} s\n'.format(totalruntime)
                statfile.write('EMPCA runtime {0:.2f} s\n\n'.format(totalruntime))

        

################################################################################

# Plot both Boolean mask and updated bitmask.

################################################################################

    if not isinstance(testsample.numstars,dict):
        totalstars = testsample.numstars
        match = np.where(testsample.data)
    elif isinstance(testsample.numstars,dict):
        totalstars = np.sum(testsample.numstars.values())
        match = (np.array([],dtype=np.int),)
        for subgroup in testsample.subgroups:
            newind = np.where(testsample.data[testsample.subgroup] == subgroup)
            match = (np.concatenate((match[0],newind[0])),)

    plt.figure(1,figsize=(16,14))
    allmasksplot = np.copy(testsample.mask[match].astype(np.float64))
    allmasksplot[np.where(testsample.mask[match]==0)] = np.nan
    plt.imshow(allmasksplot,aspect = 7214./totalstars,interpolation='nearest',cmap = 'viridis')
    plt.ylim(0,totalstars)
    plt.colorbar()
    if label != 0:
        plt.savefig('./{0}/test2_mask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
    elif label == 0:
        plt.savefig('./{0}/test2_mask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
    plt.close()

    plt.figure(2,figsize=(16,14))
    allbitmasksplot = np.copy(testsample.bitmask[match]).astype(np.float64)
    allbitmasksplot[np.where(testsample.bitmask[match]==0)] = np.nan
    plt.imshow(np.log2(allbitmasksplot),aspect = 7214./totalstars,interpolation='nearest',cmap = 'viridis')
    plt.ylim(0,totalstars)
    plt.colorbar()
    if label != 0:
        plt.savefig('./{0}/test2_bitmask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
    elif label == 0:
        plt.savefig('./{0}/test2_bitmask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
    plt.close()

    plt.figure(3,figsize=(16,14))
    SNRplot = testsample.specs[match]/testsample.errs[match]
    SNRplot[np.where(SNRplot.mask!=0)] = np.nan
    plt.imshow(SNRplot,aspect = 7214./totalstars,interpolation='nearest',cmap = 'viridis')
    plt.ylim(0,totalstars)
    plt.colorbar()
    if label != 0:
        plt.savefig('./{0}/test2_SNR_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
    elif label == 0:
        plt.savefig('./{0}/test2_SNR_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
    plt.close()

################################################################################

# Re-run again, this time to check results of fitting.

################################################################################

if run3:

    if verbose:
        print '\n\nTesting fitting routines - produces false data sets and fit\n'

    
    # Remove possible cached data.
    if label != 0:
        os.system('rm ./{0}/pickles/*{1}*{2}*{3}*'.format(samptype,label,up,low))
    elif label == 0:
        os.system('rm ./{0}/pickles/*'.format(samptype))

    # Initialize the sample
    testsample,runtime = timeIt(Sample,samptype,order=order,cross=cross,label=label,up=up,low=low,subgroup_type=subgroup_info[0],subgroup_lis=subgroup_info[1:],fontsize=10)
    if verbose:
        print '\nInitialization runtime {0:.2f} s\n'.format(runtime)
        print 'Number of stars {0}\n'.format(testsample.numstars)


    if label != 0:
        statfilename = './{0}/test-statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low)
    elif label == 0:
         statfilename = './{0}/test-statfile_order{1}_seed{2}_cross{3}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross)
    try:
        statfile.write('################################################################################\n')
        statfile.write('Run 3 - Test Pixel Fitting (uses model data set and save files)\n')
        statfile.write('################################################################################\n\n')
    except NameError:
        statfile = open(statfilename,'w+')
        statfile.write('################################################################################\n')
        statfile.write('Run 3 - Test Pixel Fitting (uses model data set and save files)\n')
        statfile.write('################################################################################\n\n')

    statfile.write('Initialization runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Number of stars {0}\n\n'.format(testsample.numstars))

    # Correct high SNR
    noneholder,runtime = timeIt(testsample.snrCorrect)
    if verbose:
        print 'SNR correction runtime {0:.2f} s\n'.format(runtime)
    statfile.write('SNR correction runtime {0:.2f} s\n\n'.format(runtime))

    # Mask low SNR
    noneholder,runtime = timeIt(testsample.snrCut)
    if verbose:
        print 'SNR cut runtime {0:.2f} s\n'.format(runtime)
    statfile.write('SNR cut runtime {0:.2f} s\n\n'.format(runtime))

    # Apply bitmask
    maskbits = bitmask.bits_set(badcombpixmask+2**15)
    noneholder,runtime = timeIt(testsample.bitmaskData,maskbits = badcombpixmask+2**15)
    if verbose:
        print 'Bitmask application runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Bitmask application runtime {0:.2f} s\n\n'.format(runtime))

    # Get independent variable arrays
    noneholder,runtime = timeIt(testsample.allIndepVars)
    if verbose:
        print 'Independent variable array generation runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Independent variable array generation runtime {0:.2f} s\n\n'.format(runtime))

    # Create a false spectra generated from independent variables using a polynomial with testparam.
    if not isinstance(testsample.allindeps,dict):
        # Choose test set of parameters and column codes.
        # These are a rounded version of the actual fit parameters at pixel 1065.
        testparam,testcolcode = testsample.pixFit(1065,testsample.allindeps[1065])
        correct_margin = abs(testparam/100.)
        for i in range(len(testsample.allindeps)):
            #testsample.specs.mask[:,i] = False
            #testsample.errs.mask[:,i] = False
            testsample.errs.data[:,i] = np.ones(len(testsample.numstars))/100.
            polycalc = pf.poly(testparam,testcolcode,testsample.allindeps[i],order = testsample.order)
            testsample.specs.data[:,i] = polycalc
            try:
                assert np.array_equal(polycalc.mask,testsample.specs.mask[:,i])
            except AssertionError:
                if verbose:
                    print 'Masks do not match at pixel {0}\n'.format(i)
                statfile.write('Masks do not match at pixel {0}\n\n'.format(i))
    #elif isinstance(testsample.allindeps,dict):
    #    for subgroup in self.subgroups:


    # Do pixel fitting
    noneholder,runtime = timeIt(testsample.allPixFit)
    if verbose:
        print 'Pixel fitting runtime {0:.2f} s'.format(runtime)
    statfile.write('Pixel fitting runtime {0:.2f} s\n'.format(runtime))
    for pix in range(len(testsample.allparams)):
        failpixs = []
        if not all(abs(testsample.allparams[pix]-testparam) < correct_margin):
            failpixs.append(pix)
    if verbose:
        print 'Failed pixel fit percentage {0:.2f}\n'.format(float(len(failpixs))/float(aspcappix))
    statfile.write('Failed pixel fit percentage {0:.2f}\n\n'.format(float(len(failpixs))/float(aspcappix)))

    # Do residual calculation
    noneholder,runtime = timeIt(testsample.allPixResiduals)
    if verbose:
        print 'Pixel residuals runtime {0:.2f} s'.format(runtime)
        print 'Maximum residual {0} '.format(np.max(testsample.residual))
    statfile.write('Pixel residuals runtime {0:.2f} s\n'.format(runtime))
    statfile.write('Maximum residual {0}\n\n'.format(np.max(testsample.residual)))

    # Gather random sigma
    noneholder,runtime = timeIt(testsample.allRandomSigma)
    if verbose:
        print 'Finding random sigma runtime {0:.2f} s\n'.format(runtime)
    statfile.write('Finding random sigma runtime {0:.2f} s\n\n'.format(runtime))


try:
    statfile.close()
except NameError:
    print 'No test runs selected, no output status file.'
