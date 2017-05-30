#!/usr/bin/env python

"""
Weighted Principal Component Analysis using Expectation Maximization

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    m = empca(data, weights, options...)

Returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )
    
For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #- but no weights or even options...
    
Stephen Bailey, Spring 2012
"""

import numpy as N
import sys
from scipy.sparse import dia_matrix
import scipy.sparse.linalg
import math

k = 1.4826 #scaling factor that makes MAD=variance in Gaussian case

def MAD(arr):
    """
    An example function that can be used to calculate the variance using the
    median absolute deviation
     Inputs
       - arr: a masked array with shape [nobs, nvar]
     Outputs
       - variance: a 1D scalar
    
    """
    return k**2*N.ma.median((arr-N.ma.median(arr))**2)[0] 

def meanMed(arr):
    """                                                                        
    An example function that can be used to calculate the variance using the   
    median absolute deviation                                                  
     Inputs                                                                    
       - arr: a masked array with shape [nobs, nvar]                           
     Outputs                                                                   
       - variance: a 1D scalar
    """

    meds = N.ma.median(arr,axis=0)
    medarr = N.tile(meds,(arr.shape[0],1))
    medsub = N.ma.median((arr-medarr)**2,axis=0)
    return k**2*N.ma.mean(medsub)
    
class Model(object):
    """
    A wrapper class for storing data, eigenvectors, and coefficients.
    
    Returned by empca() function.  Useful member variables:
      Inputs: 
        - eigvec [nvec, nvar]
        - data   [nobs, nvar]
        - weights[nobs, nvar]
      
      Calculated from those inputs:
        - coeff  [nobs, nvec] - coeffs to reconstruct data using eigvec
        - model  [nobs, nvar] - reconstruction of data using eigvec,coeff
    
    Not yet implemented: eigenvalues, mean subtraction/bookkeeping
    """
    def __init__(self, eigvec, data, weights,varfunc=N.ma.var):
        """
        Create a Model object with eigenvectors, data, and weights.
        
        Dimensions:
          - eigvec [nvec, nvar]  = [k, j]
          - data   [nobs, nvar]  = [i, j]
          - weights[nobs, nvar]  = [i, j]
          - coeff  [nobs, nvec]  = [i, k]        
        """
        self.eigvec = eigvec
        self.nvec = eigvec.shape[0]
        self.varfunc = varfunc
        self.set_data(data, weights)

        
    def set_data(self, data, weights):
        """
        Assign a new data[nobs,nvar] and weights[nobs,nvar] to use with
        the existing eigenvectors.  Recalculates the coefficients and
        model fit.
        """
        self.data = data
        self.weights = weights

        self.weighted_data = self.data*self.weights
        self.weighted_data[self.weighted_data!=0]/=N.sqrt(self.weights[self.weighted_data!=0]**2)
        self.datamean = N.mean(self.weighted_data,axis=0)
        self.data -= self.datamean
        self.meanstack = N.tile(self.datamean,(self.data.shape[0],1))

        self.nobs = data.shape[0]
        self.nvar = data.shape[1]
        self.coeff = N.zeros( (self.nobs, self.nvec) )
        self.model = N.zeros( self.data.shape )
        
        #- Calculate degrees of freedom
        ii = N.where(self.weights>0)
        self.dof = self.data[ii].size - self.eigvec.size  - self.nvec*self.nobs
        
        #- Cache variance of unmasked data
        self._unmasked = ii
        self._unmasked_data_var = self.varfunc(self.mask_data(self.data))
        self.solve_coeffs()

    def mask_data(self,arr):
        return N.ma.masked_array(arr,mask=self.weights==0)
        
    def solve_coeffs(self):
        """
        Solve for c[i,k] such that data[i] ~= Sum_k: c[i,k] eigvec[k]
        """
        for i in range(self.nobs):
            #- Only do weighted solution if really necessary
            if N.any(self.weights[i] != self.weights[i,0]):
                self.coeff[i] = _solve(self.eigvec.T, self.data[i], self.weights[i])
            else:
                self.coeff[i] = N.dot(self.eigvec, self.data[i])
            
        self.solve_model()
            
    def solve_eigenvectors(self, smooth=None):
        """
        Solve for eigvec[k,j] such that data[i] = Sum_k: coeff[i,k] eigvec[k]
        """

        #- Utility function; faster than numpy.linalg.norm()
        def norm(x):
            return N.sqrt(N.dot(x, x))
            
        #- Make copy of data so we can modify it
        data = self.data.copy()

        #- Solve the eigenvectors one by one
        for k in range(self.nvec):

            c = N.tile(self.coeff[:,k],(self.nvar,1)).T
            cw = c*self.weights
            numer = N.sum(data*cw,axis=0)
            denom = N.sum(c*cw,axis=0)
            self.eigvec[k] = numer/denom
            

            ##- Can we compact this loop into numpy matrix algebra?
            #c = self.coeff[:, k]
            #for j in range(self.nvar):
            #    w = self.weights[:, j]
            #    x = data[:, j]
            #    cw = c*w
            #    self.eigvec[k, j] = x.dot(cw) / c.dot(cw)
                                                
            if smooth is not None:
                self.eigvec[k] = smooth(self.eigvec[k])

            #- Remove this vector from the data before continuing with next
            data -= N.outer(self.coeff[:,k], self.eigvec[k])    

        #- Renormalize and re-orthogonalize the answer
        self.eigvec[0] /= norm(self.eigvec[0])
        for k in range(1, self.nvec):
            for kx in range(0, k):
                c = N.dot(self.eigvec[k], self.eigvec[kx])
                self.eigvec[k] -=  c * self.eigvec[kx]
                    
            self.eigvec[k] /= norm(self.eigvec[k])
        self.rank_eigvec()

        #- Recalculate model
        self.solve_model()
           
    def rank_eigvec(self):
        """
        Sort eigenvectors by the fraction of variance they explain.
        """
        vars = N.zeros(len(self.eigvec))
        for i in range(len(self.eigvec)):
            vars[i] = self.R2vec(i)
        order = N.argsort(vars)[::-1]
        self.eigvec = self.eigvec[order]
        self.coeff = self.coeff.T[order].T

    def check_orthogonality(self):
        """
        Check that all eigenvectors are orthogonal
        """
        # Create an array to hold results for pairs not including (i,i)
        self.orthog_check = zeros(len(self.eigvec)**2-len(self.eigvec))
        k = 0
        pairs = []
        for i in range(len(self.eigvec)):
            for j in range(len(self.eigvec)):
                if i!=j:
                    self.orthog_check[k] = np.dot(self.eigvec[i],self.eigvec[j])
                    self.orthog_check_pairs.append('({0},{1})'.format(i+1,j+1))
                    k+=1
        return np.sum(self.orthog_check)

    def solve_model(self):
        """
        Uses eigenvectors and coefficients to model data
        """
        for i in range(self.nobs):
            self.model[i] = self.eigvec.T.dot(self.coeff[i])
                       
    def chi2(self):
        """
        Returns sum( (model-data)^2 / weights )
        """
        delta = (self.model - self.data) * N.sqrt(self.weights)
        return N.sum(delta**2)
        
    def rchi2(self):
        """
        Returns reduced chi2 = chi2/dof
        """
        return self.chi2() / self.dof
        
    def _model_vec(self, i, addmean=False):
        """Return the model using just eigvec i"""
        if not addmean:
            return N.outer(self.coeff[:, i], self.eigvec[i])
        elif addmean:
            return N.outer(self.coeff[:, i], self.eigvec[i])+self.meanstack
        
    def eigval(self,nvec=None):

        if nvec is None:
            return 0
        else:
            mx = N.zeros(self.data.shape)
            mx_1 = N.zeros(self.data.shape)
            c = 0
            for i in range(nvec):
                mx += self._model_vec(i)
                if c < nvec-1:
                    mx_1 += self._model_vec(i)
                c+=1
            d = mx - self.data
            d_1 = mx_1 - self.data
            Vdatai = self.varfunc(self.mask_data(d))
            Vdata_1 = self.varfunc(self.mask_data(d_1))
            return Vdata_1-Vdatai

    def R2vec(self, i):
        """
        Return fraction of data variance which is explained by vector i.

        Notes:
          - Does *not* correct for degrees of freedom.
        """
        
        d = self._model_vec(i) - self.data
        return 1.0 - self.varfunc(self.mask_data(d))/self._unmasked_data_var
                
    def R2(self, nvec=None):
        """
        Return fraction of data variance which is explained by the first
        nvec vectors.  Default is R2 for all vectors.
        
        Notes:
          - Does *not* correct for degrees of freedom.
        """
        if nvec is None:
            mx = self.model
        else:            
            mx = N.zeros(self.data.shape)
            for i in range(nvec):
                mx += self._model_vec(i)
            
        d = mx - self.data

        # Only consider R2 for unmasked data
        return 1.0 - self.varfunc(self.mask_data(d))/self._unmasked_data_var         
       
def _random_orthonormal(nvec, nvar, seed=1):
    """
    Return array of random orthonormal vectors A[nvec, nvar] 

    Doesn't protect against rare duplicate vectors leading to 0s
    """

    if seed is not None:
        N.random.seed(seed)
        
    A = N.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        A[i] /= N.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= N.dot(A[j], A[i]) * A[j]
            A[i] /= N.linalg.norm(A[i])

    return A

def _solve(A, b, w):
    """
    Solve Ax = b with weights w; return x
    
    A : 2D array
    b : 1D array length A.shape[0]
    w : 1D array same length as b
    """
    
    b = A.T.dot( w*b )
    A = A.T.dot( (A.T * w).T )

    if isinstance(A, scipy.sparse.spmatrix):
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        x = N.linalg.lstsq(A, b)[0]
        
    return x

    
#-------------------------------------------------------------------------

def empca(data, weights=None, deltR2=0,niter=25, nvec=5, smooth=0, randseed=1, silent=False, varfunc=N.var):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights
    
    Input:
      - data[nobs, nvar]
      - weights[nobs, nvar]
      
    Optional:
      - deltR2   : difference between consecutive R2 at which to halt iteration
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - randseed : random number generator seed; None to not re-initialize
      - mad      : option to calculate R2 with median absolute deviation of data rather than variance
    
    Returns Model object
    """

    if weights is None:
        weights = N.ones(data.shape)

    if smooth>0:
        smooth = SavitzkyGolay(width=smooth)
    else:
        smooth = None

    #- Basic dimensions
    nobs, nvar = data.shape
    assert data.shape == weights.shape

    #- degrees of freedom for reduced chi2
    ii = N.where(weights > 0)
    dof = data[ii].size - nvec*nvar - nvec*nobs 

    #- Starting random guess
    eigvec = _random_orthonormal(nvec, nvar, seed=randseed)
    
    model = Model(eigvec, data, weights,varfunc=varfunc)
    model.solve_coeffs()
    
    if not silent:
        # print "       iter    chi2/dof     drchi_E     drchi_M   drchi_tot       R2            rchi2"
        print "       iter        R2             rchi2"
    
    R2_old = 0.
    for k in range(niter):
        model.solve_coeffs()
        model.solve_eigenvectors(smooth=smooth)
        R2_new = model.R2()
        R2diff = N.fabs(R2_new-R2_old)
        R2_old = R2_new
        if not silent:
            print 'EMPCA %2d/%2d  %15.8f %15.8f' % \
                (k+1, niter, model.R2(), model.rchi2())
            sys.stdout.flush()
        if R2diff < deltR2:
            break

    #- One last time with latest coefficients
    model.solve_coeffs()

    if not silent:
        print "R2:", model.R2()
    
    return model

def classic_pca(data, nvec=None):
    """
    Perform classic SVD-based PCA of the data[obs, var].
    
    Returns Model object
    """
    u, s, v = N.linalg.svd(data)
    if nvec is None:
        m = Model(v, data, N.ones(data.shape))    
    else:
        m = Model(v[0:nvec], data, N.ones(data.shape))
    return m

def lower_rank(data, weights=None, niter=25, nvec=5, randseed=1):
    """
    Perform iterative lower rank matrix approximation of data[obs, var]
    using weights[obs, var].
    
    Generated model vectors are not orthonormal and are not
    rotated/ranked by ability to model the data, but as a set
    they are good at describing the data.
    
    Optional:
      - niter : maximum number of iterations to perform
      - nvec  : number of vectors to solve
      - randseed : rand num generator seed; if None, don't re-initialize
    
    Returns Model object
    """
    
    if weights is None:
        weights = N.ones(data.shape)
    
    nobs, nvar = data.shape
    P = _random_orthonormal(nvec, nvar, seed=randseed)
    C = N.zeros( (nobs, nvec) )
    ii = N.where(weights > 0)
    dof = data[ii].size - P.size - nvec*nobs 

    print "iter     dchi2       R2             chi2/dof"

    oldchi2 = 1e6*dof
    for blat in range(niter):
        #- Solve for coefficients
        for i in range(nobs):
            #- Convert into form b = A x
            b = data[i]              #- b[nvar]
            A = P.T                  #- A[nvar, nvec]
            w = weights[i]           #- w[nvar]            
            C[i] = _solve(A, b, w)   #- x[nvec]
                        
        #- Solve for eigenvectors
        for j in range(nvar):
            b = data[:, j]           #- b[nobs]
            A = C                    #- A[nobs, nvec]
            w = weights[:, j]        #- w[nobs]
            P[:,j] = _solve(A, b, w) #- x[nvec]
            
        #- Did the model improve?
        model = C.dot(P)
        delta = (data - model) * N.sqrt(weights)
        chi2 = N.sum(delta[ii]**2)
        diff = data-model
        R2 = 1.0 - N.var(diff[ii]) / N.var(data[ii])
        dchi2 = (chi2-oldchi2)/oldchi2   #- fractional improvement in chi2
        flag = '-' if chi2<oldchi2 else '+'
        print '%3d  %9.3g  %15.8f %15.8f %s' % (blat, dchi2, R2, chi2/dof, flag)
        oldchi2 = chi2

    #- normalize vectors
    for k in range(nvec):
        P[k] /= N.linalg.norm(P[k])

    m = Model(P, data, weights)
    print "R2:", m.R2()

    #- Rotate basis to maximize power in lower eigenvectors
    #--> Doesn't work; wrong rotation
    # u, s, v = N.linalg.svd(m.coeff, full_matrices=True)
    # eigvec = N.zeros(m.eigvec.shape)
    # for i in range(m.nvec):
    #     for j in range(s.shape[0]):
    #         eigvec[i] += v[i,j] * m.eigvec[j]
    # 
    #     eigvec[i] /= N.linalg.norm(eigvec[i])
    # 
    # m = Model(eigvec, data, weights)
    # print m.R2()
    
    return m

class SavitzkyGolay(object):
    """
    Utility class for performing Savitzky Golay smoothing
    
    Code adapted from http://public.procoders.net/sg_filter/sg_filter.py
    """
    def __init__(self, width, pol_degree=3, diff_order=0):
        self._width = width
        self._pol_degree = pol_degree
        self._diff_order = diff_order
        self._coeff = self._calc_coeff(width//2, pol_degree, diff_order) 

    def _calc_coeff(self, num_points, pol_degree, diff_order=0):
    
        """
        Calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf
    
        num_points   means that 2*num_points+1 values contribute to the
                     smoother.
    
        pol_degree   is degree of fitting polynomial
    
        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first 
                                                 derivative of function.
                     and so on ...
        """
    
        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....
    
        x = N.arange(-num_points, num_points+1, dtype=int)
        monom = lambda x, deg : math.pow(x, deg)
    
        A = N.zeros((2*num_points+1, pol_degree+1), float)
        for i in range(2*num_points+1):
            for j in range(pol_degree+1):
                A[i,j] = monom(x[i], j)
            
        # calculate diff_order-th row of inv(A^T A)
        ATA = N.dot(A.transpose(), A)
        rhs = N.zeros((pol_degree+1,), float)
        rhs[diff_order] = (-1)**diff_order
        wvec = N.linalg.solve(ATA, rhs)
    
        # calculate filter-coefficients
        coeff = N.dot(A, wvec)
    
        return coeff
    
    def __call__(self, signal):
        """
        Applies Savitsky-Golay filtering
        """
        n = N.size(self._coeff-1)/2
        res = N.convolve(signal, self._coeff)
        return res[n:-n]


def _main():
    N.random.seed(1)
    nobs = 100
    nvar = 200
    nvec = 3
    data = N.zeros(shape=(nobs, nvar))

    #- Generate data
    x = N.linspace(0, 2*N.pi, nvar)
    for i in range(nobs):
        for k in range(nvec):
            c = N.random.normal()
            data[i] += 5.0*nvec/(k+1)**2 * c * N.sin(x*(k+1))

    #- Add noise
    sigma = N.ones(shape=data.shape)
    for i in range(nobs/10):
        sigma[i] *= 5
        sigma[i, 0:nvar/4] *= 5

    weights = 1.0 / sigma**2    
    noisy_data = data + N.random.normal(scale=sigma)

    print "Testing empca"
    m0 = empca(noisy_data, weights, niter=20)
    
    print "Testing lower rank matrix approximation"
    m1 = lower_rank(noisy_data, weights, niter=20)
    
    print "Testing classic PCA"
    m2 = classic_pca(noisy_data)
    print "R2", m2.R2()
    
    try:
        import pylab as P
    except ImportError:
        print >> sys.stderr, "pylab not installed; not making plots"
        sys.exit(0)
        
    P.subplot(311)
    for i in range(nvec): P.plot(m0.eigvec[i])
    P.ylim(-0.2, 0.2)
    P.ylabel("EMPCA")
    P.title("Eigenvectors")
    
    P.subplot(312)
    for i in range(nvec): P.plot(m1.eigvec[i])
    P.ylim(-0.2, 0.2)
    P.ylabel("Lower Rank")
    
    P.subplot(313)
    for i in range(nvec): P.plot(m2.eigvec[i])
    P.ylim(-0.2, 0.2)
    P.ylabel("Classic PCA")
    
    P.show()
        
if __name__ == '__main__':
    _main()
    






    
