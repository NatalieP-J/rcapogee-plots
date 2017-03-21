# rcapogee-plots

## Purpose

This code analyzes the chemical composition of stars from APOGEE in spectral space. It uses polynomial fits to remove distortion of stellar spectra due effective temperature, surface gravity and overall metallicity [Fe/H]. After subtracting these fits, the resulting spectra are processed through dimension-reducing algorithm Expectation Maximized Principal Component Analysis (EMPCA) to find the dimensions of most importance for distinguishing spectra.

## Installation

Clone this repository and add it to your $PYTHONPATH variable:

`prompt$ export PYTHONPATH=$PYTHONPATH:<path to repository>/rcapogee-plots`

This repository requires the following packages: [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/), [tqdm](https://pypi.python.org/pypi/tqdm), [sklearn](http://scikit-learn.org/stable/), [apogee](https://github.com/jobovy/apogee), [astropy](http://www.astropy.org/), [scipy](https://www.scipy.org/), [statsmodels](http://statsmodels.sourceforge.net/)

## How to Use rcapogee-plots

In general, we wish to create an empca_residuals object then use its associated functions. Here's a basic example to get you started.
From any directory, open python.

```python
>>>: from empca_residuals import empca_residuals,maskFilter
>>>: rcsample = empca_residuals('apogee','red_clump',maskFilter,datadir='.')

```

You've now selected a subsample of the red clump stars from APOGEE's DR12 according to a temperature cut. This step has read in all the spectra and stellar parameters and created an output directory called `red_clump_12_TEFF_up4900.0_lo4800.0/bm4351` in the working directory, where bm4351 indicates which bits were masked on. Let's try fitting those stars.

```python
>>>: fitparams = rcsample.findFit(1000)
```

This gives us the best polynomial fit values for pixel 1000, as well as coefficients for each term in the polynomial and their uncertainties (default choice for polynomial degree choice is 2). An APOGEE spectrum has about 7000 pixels, so let's run all fits and compute the residuals.

```python
>>>: rcsample.findResiduals()
```

Once we've computed the residuals, we can run EMPCA on the sample. This step may take up to half an hour.

```python
>>>: rcsample.pixelEMPCA(nvecs=20,savename='test.pkl)
```

Upon completion, the output directory should now contain a file called `test.pkl` (`red_clump_12_TEFF_up4900.0_lo4800.0/bm4351/test.pkl`). This file stores statistical information about the results of the EMPCA. We can recover this information by reloading it, although it is still stored in our rcsample object.

```python
>>>: import access_spectrum as acs
>>>: EMPCA_model = acs.pklread('red_clump_12_TEFF_up4900.0_lo4800.0/bm4351/test.pkl')
>>>: attributes = [a for a in dir(EMPCA_model) if not a.startswith('__')]
>>>: print attributes
['coeff','correction','data','eigvals','eigvec','R2Array','R2noise','Vdata','Vnoise','savename']
```

These attributes all also exist within our original object, accessible via `rcsample.empcaModelWeight` (e.g, `rcsample.empcaModelWeight.Rnoise`).
