# spectralspace
### Author: Natalie Price-Jones
### Publications: [Price-Jones & Bovy (2017)](www.astro.utoronto.ca/~price-jones/drafts/rc_dim.pdf)

## Purpose

This code reduces the dimensionality of APOGEE H-band spectra ([Majewski et. al. (2015)](https://arxiv.org/abs/1509.05420)). It uses polynomial fits to remove distortion of stellar spectra due effective temperature, surface gravity and overall metallicity [Fe/H]. After subtracting these fits, the resulting spectra are processed through dimension-reducing algorithm Expectation Maximized Principal Component Analysis (EMPCA - [Bailey (2012)](https://arxiv.org/abs/1208.4122)) to find the dimensions of most importance for distinguishing spectra.

## Installation

Clone this repository. Note that the repository has a significant history and so total size is 125 MB. To reduce download size, consider  

`git clone --depth=1 https://github.com/npricejones/spectralspace.git`

Navigate into the repository to install with:

`python setup.py install`

with possible option for local installation only:

`python setup.py install --prefix=<some local directory>`

This repository requires the following packages: [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/), [tqdm](https://pypi.python.org/pypi/tqdm), [sklearn](http://scikit-learn.org/stable/), [apogee](https://github.com/jobovy/apogee), [astropy](http://www.astropy.org/), [scipy](https://www.scipy.org/), [statsmodels](http://statsmodels.sourceforge.net/), [galpy](https://github.com/jobovy/galpy), [isodist](https://github.com/jobovy/isodist))

If you wish to try the example notebooks, you'll also need [jupyter](http://jupyter.org)

## How to Use spectralspace

In general, we wish to create an empca_residuals object then use its associated functions. Here's a basic example to get you started.
From any directory, open python.

```python
>>> from spectralspace.analysis.empca_residuals import empca_residuals,maskFilter
>>> rcsample = empca_residuals('apogee','red_clump',maskFilter,datadir='.')

```

You've now selected a subsample of the red clump stars from APOGEE's DR12 according to a temperature cut. This step has read in all the spectra and stellar parameters and created an output directory called `red_clump_12_TEFF_up4900.0_lo4800.0/bm4351` in the `datadir` (defaults to working directory), where bm4351 indicates which bits are used in the default mask, defined by `maskFilter`. Let's try fitting those stars.

```python
>>> fitparams = rcsample.findFit(1000)
```

This gives us the best polynomial fit values for pixel 1000, as well as coefficients for each term in the polynomial and their uncertainties (default choice for polynomial degree choice is 2). An APOGEE spectrum has about 7000 pixels, so let's run all fits and compute the residuals.

```python
>>> rcsample.findResiduals()
```

Once we've computed the residuals, we can run EMPCA with 20 principal components on the sample. This step may take up to half an hour.

```python
>>> rcsample.pixelEMPCA(nvecs=20,savename='test.pkl)
```

Upon completion, the output directory should now contain a file called `test.pkl` (`red_clump_12_TEFF_up4900.0_lo4800.0/bm4351/test.pkl`). This file stores statistical information about the results of the EMPCA. We can recover this information by reloading it, although it is still stored in our rcsample object.

```python
>>> import spectralspace.sample.access_spectrum as acs
>>> EMPCA_model = acs.pklread('red_clump_12_TEFF_up4900.0_lo4800.0/bm4351/test.pkl')
>>> attributes = [a for a in dir(EMPCA_model) if not a.startswith('__')]
>>> print attributes
['coeff','correction','data','eigvals','eigvec','R2Array','R2noise','Vdata','Vnoise','savename']
```

These attributes all also exist within our original object, accessible via `rcsample.empcaModelWeight` (e.g, `rcsample.empcaModelWeight.Rnoise`).

For more detailed and specific examples, see the `RC_dimensionality_plots.ipynb` notebook in the examples folder.
