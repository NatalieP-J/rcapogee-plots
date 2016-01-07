"""

Usage:
test_residuals [-hvpglfe] [-i LABEL] [-u UPLIM] [-d LOWLIM] [-s SAMPLETYPE]

Sample call:

test_residuals -gel -i FE_H -u -0.1 -d -0.105 -s red_clump

Options:
    -h, --help
    -v, --verbose
    -p, --pixplot                       Option to turn on fit plots at each pixel.
    -g, --generate                      Option to run first sequence (generate everything from scratch)
    -e, --empca                         Option to run empca.
    -l,	--loadin                        Option to run second sequence (load from file where possible)
    -f, --fittest                       Option to run third sequence (test polynomial fits)
    -i LABEL, --indep LABEL             A string with a label in which to crop the sample 
                                        [default: 0]
    -u UPLIM, --upper UPLIM             An upper limit for the sample crop 
                                        [default: 0]
    -d LOWLIM, --lower LOWLIM           A lower limit for the sample crop 
                                        [default: 0]
    -s SAMPTYPE, --samptype SAMPTYPE    The type of sample to run 
                                        [default: 'clusters']

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

arguments = docopt.docopt(__doc__)

verbose = arguments['--verbose']
pixplot = arguments['--pixplot']
empcarun = arguments['--empca']
run1 = arguments['--generate']
run2 = arguments['--loadin']
run3 = arguments['--fittest']

label = arguments['--indep']
up = float(arguments['--upper'])
low = float(arguments['--lower'])
samptype = arguments['--samptype']

if label == '0':
	label = 0

################################################################################

# Code used to test residuals.py and confirm all functions are in working order.

################################################################################

def timeIt(fn,*args,**kwargs):
	"""
	A basic function to time how long another function takes to run.

	fn: 	Function to time.

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
	testsample,runtime = timeIt(Sample,samptype,label=label,up=up,low=low,fontsize=10)

	if label != 0:
		statfilename = './{0}/statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low)
	elif label == 0:
		 statfilename = './{0}/statfile_order{1}_seed{2}_cross{3}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross)
	statfile = open(statfilename,'w+')

	if verbose:
		print '\nInitialization runtime {0:.2f} s'.format(runtime)
		print 'Number of stars {0}\n'.format(testsample.numstars)
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
		assert np.array_equal(np.where(testsample.specs.data/testsample.errs.data < 50)[0],np.where(testsample.specs.mask == True)[0])
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
	for i in range(len(testsample.allindeps)):
		for indep in testsample.allindeps[i]:
			try:
				assert np.array_equal(testsample.specs.mask[:,i],indep.mask)
			except AssertionError:
				if verbose:
					print 'Independent variables improperly masked.\n'
				statfile.write('Independent variables improperly masked.\n\n')

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
		empcamodel,runtime = timeIt(empca,testsample.residual.T)
		empcamodel_weight,runtime = timeIt(empca,testsample.residual.T,1./(testsample.errs**2))
		if verbose:
			print 'EMPCA runtime {0:.2f} s\n'.format(runtime)
		statfile.write('EMPCA runtime {0:.2f} s\n\n'.format(runtime))
		for elem in tophats.keys():
			window = windowPeaks[elem]
			labels = windowPeaks[elem][0]
			plot1 = empcamodel.eigvec[0][window]
			plot2 = empcamodel_weight.eigvec[0][window]
			plt.figure(1,figsize=(16,14))
			plt.plot(plot1,'o',label = 'Unweighted EMPCA')
			plt.plot(plot2,'o',label = 'Weighted EMPCA')
			plt.axhline(0,color='red')
			slicestep = int(10**np.round((np.log10(len(labels))-1)))
			plt.xticks(range(len(labels))[::slicestep], labels[::slicestep])
			plt.xlim(0,len(labels))
			plt.title('First Eigenvector - {0}'.format(elem))
			plt.legend(loc='best')
			if label != 0:
				plt.savefig('./{0}/eigvec1_{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(testsample.type, elem,testsample.order,testsample.seed,testsample.cross,label,up,low))
			elif label == 0:
				plt.savefig('./{0}/eigvec1_{1}_order{2}_seed{3}_cross{4}.png'.format(testsample.type, elem,testsample.order,testsample.seed,testsample.cross))
			plt.close()

################################################################################

# Plot both Boolean mask and updated bitmask.

################################################################################

	plt.figure(1,figsize=(16,14))
	allmasksplot = np.copy(testsample.mask.astype(np.float64))
	allmasksplot[np.where(testsample.mask==0)] = np.nan
	plt.imshow(allmasksplot,aspect = 7214./testsample.numstars,interpolation='nearest',cmap = 'viridis')
	plt.colorbar()
	if label != 0:
		plt.savefig('./{0}/test1_mask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
	elif label == 0:
		plt.savefig('./{0}/test1_mask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
	plt.close()

	plt.figure(2,figsize=(16,14))
	allbitmasksplot = np.copy(testsample.bitmask).astype(np.float64)
	allbitmasksplot[np.where(testsample.bitmask==0)] = np.nan
	plt.imshow(np.log2(allbitmasksplot),aspect = 7214./testsample.numstars,interpolation='nearest',cmap = 'viridis')
	plt.colorbar()
	if label != 0:
		plt.savefig('./{0}/test1_bitmask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
	elif label == 0:
		plt.savefig('./{0}/test1_bitmask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
	plt.close()

################################################################################

# Re-run all steps to check that caching data reduces run time.

################################################################################

if run2:

	if verbose:
		print '\n\nLoading from saved files\n'

	# Initialize the sample
	testsample,runtime = timeIt(Sample,'red_clump',label=label,up=up,low=low,fontsize=10)
	if verbose:
		print '\nInitialization runtime {0:.2f} s'.format(runtime)
		print 'Number of stars {0}\n'.format(testsample.numstars)

	if label != 0:
		statfilename = './{0}/statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low)
	elif label == 0:
		 statfilename = './{0}/statfile_order{1}_seed{2}_cross{3}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross)
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
		assert np.array_equal(np.where(testsample.specs.data/testsample.errs.data < 50)[0],np.where(testsample.specs.mask == True)[0])
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
	for i in range(len(testsample.allindeps)):
		for indep in testsample.allindeps[i]:
			try:
				assert np.array_equal(testsample.specs.mask[:,i],indep.mask)
			except AssertionError:
				if verbose:
					print 'Independent variables improperly masked.\n'
				statfile.write('Independent variables improperly masked.\n\n')

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

	# Get empca results
	if empcarun:
		empcamodel,runtime = timeIt(empca,testsample.residual.T)
		empcamodel_weight,runtime = timeIt(empca,testsample.residual.T,1./(testsample.errs**2))
		if verbose:
			print 'EMPCA runtime {0:.2f} s\n'.format(runtime)
		statfile.write('EMPCA runtime {0:.2f} s\n\n'.format(runtime))
		for elem in tophats.keys():
			window = windowPeaks[elem]
			labels = windowPeaks[elem][0]
			plot1 = empcamodel.eigvec[0][window]
			plot2 = empcamodel_weight.eigvec[0][window]
			plt.figure(1,figsize=(16,14))
			plt.plot(plot1,'o',label = 'Unweighted EMPCA')
			plt.plot(plot2,'o',label = 'Weighted EMPCA')
			plt.axhline(0,color='red')
			slicestep = int(10**np.round((np.log10(len(labels))-1)))
			plt.xticks(range(len(labels))[::slicestep], labels[::slicestep])
			plt.xlim(0,len(labels))
			plt.title('First Eigenvector - {0}'.format(elem))
			plt.legend(loc='best')
			if label != 0:
				plt.savefig('./{0}/eigvec1_{1}_order{2}_seed{3}_cross{4}_{5}_u{6}_d{7}.png'.format(testsample.type, elem,testsample.order,testsample.seed,testsample.cross,label,up,low))
			elif label == 0:
				plt.savefig('./{0}/eigvec1_{1}_order{2}_seed{3}_cross{4}.png'.format(testsample.type, elem,testsample.order,testsample.seed,testsample.cross))
			plt.close()

################################################################################

# Plot both Boolean mask and updated bitmask.

################################################################################

	plt.figure(1,figsize=(16,14))
	allmasksplot = np.copy(testsample.mask.astype(np.float64))
	allmasksplot[np.where(testsample.mask==0)] = np.nan
	plt.imshow(allmasksplot,aspect = 7214./testsample.numstars,interpolation='nearest',cmap = 'viridis')
	plt.colorbar()
	if label != 0:
		plt.savefig('./{0}/test2_mask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
	elif label == 0:
		plt.savefig('./{0}/test2_mask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
	plt.close()

	plt.figure(2,figsize=(16,14))
	allbitmasksplot = np.copy(testsample.bitmask).astype(np.float64)
	allbitmasksplot[np.where(testsample.bitmask==0)] = np.nan
	plt.imshow(np.log2(allbitmasksplot),aspect = 7214./testsample.numstars,interpolation='nearest',cmap = 'viridis')
	plt.colorbar()
	if label != 0:
		plt.savefig('./{0}/test2_bitmask_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low))
	elif label == 0:
		plt.savefig('./{0}/test2_bitmask_order{1}_seed{2}_cross{3}.png'.format(testsample.type, testsample.order,testsample.seed,testsample.cross))
	plt.close()

################################################################################

# Re-run again, this time to check results of fitting.

################################################################################

if run3:

	if verbose:
		print '\n\nTesting fitting routines - produces false data sets and fit\n'

	os.system('rm ./red_clump/pickles/*')

	# Initialize the sample
	testsample,runtime = timeIt(Sample,'red_clump',label=label,up=up,low=low,fontsize=10)
	if verbose:
		print '\nInitialization runtime {0:.2f} s\n'.format(runtime)
		print 'Number of stars {0}\n'.format(testsample.numstars)


	if label != 0:
		statfilename = './{0}/statfile_order{1}_seed{2}_cross{3}_{4}_u{5}_d{6}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross,label,up,low)
	elif label == 0:
		 statfilename = './{0}/statfile_order{1}_seed{2}_cross{3}.txt'.format(testsample.type, testsample.order,testsample.seed,testsample.cross)
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

	# Choose test set of parameters and column codes.
	# These are a rounded version of the actual fit parameters at pixel 1065.
	testparam,testcolcode = testsample.pixFit(1065)
	correct_margin = abs(testparam/100.)

	# Create a false spectra generated from independent variables using a polynomial with testparam.
	for i in range(len(testsample.allindeps)):
		#testsample.specs.mask[:,i] = False
		#testsample.errs.mask[:,i] = False
		#testsample.errs.data[:,i] = np.ones(len(testsample.numstars))/100.
		polycalc = pf.poly(testparam,testcolcode,testsample.allindeps[i],order = testsample.order)
		testsample.specs.data[:,i] = polycalc
		try:
			assert np.array_equal(polycalc.mask,testsample.specs.mask[:,i])
		except AssertionError:
			if verbose:
				print 'Masks do not match at pixel {0}\n'.format(i)
			statfile.write('Masks do not match at pixel {0}\n\n'.format(i))

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

statfile.close()
