import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def makematrix(x,order):
    """
    Creates a matrix of the independent variable(s) for use in linear regression.
    
    x:       array of independent variable 
             (may be tuple containing arrays of multiple variables)
    order:   order of polynomial to fit
    

    Returns a matrix constructed from the independent variable(s).
    """
    if isinstance(x,tuple):
        nindeps = len(x)
        X = np.empty((len(x[0]),order*nindeps+1))
        X[:,0] = x[0]**0
        i = 1
        while i < order*nindeps+1:
            for o in range(1,order+1):
                for n in range(nindeps):
                    X[:,i] = x[n]**o
                    i+=1
    elif isinstance(x,(list,np.ndarray)):
        X = np.empty((len(x),order+1))
        for o in range(order+1):
            X[:,o] = x**o
    X = np.matrix(X)
    return X

def regfit(x,y,err = 0,order = 1):
    """
    Fits a (nD-)polynomial of specified order with independent values given in x,
    given dependent values in y.
    
    x:       array of independent variable 
             (may be tuple containing arrays of multiple variables)   
    y:       array of dependent variable (must have same shape as array in x)
    order:   order of polynomial to fit (kwarg, default = 1)
    
    Returns the polynomial coefficents ascending from 0th order. In the case of
    multiple independent variables, returns coefficients at each polynomial order
    in order of the variables listed in x.
    """
    X = makematrix(x,order)
    if isinstance(err,(float,int)):
        return np.array(np.linalg.inv(X.T*X)*X.T*np.matrix(y).T)
    elif isinstance(err,(list,np.ndarray)):
        cov = np.diag(err**2)
        icov = np.linalg.inv(cov)
        return np.array(np.linalg.inv(X.T*icov*X)*(X.T*icov*np.matrix(y).T))

def poly(p,x,order = 1):
    """
    For a given set of polynomial coefficients ascending from
    0th order, and a independent variables, returns polynomial.
    
    p:       coefficients of the polynomial in ascending order
    x:       array of independent variable
            (may be tuple containing arrays of multiple variables)
    order:   order of polynomial (kwarg, default = 1)
    
    Returns an array of polynomial values.
    """
    if isinstance(x,tuple):
        nindeps = len(x)
        order = (len(p)-1)/nindeps
        y = np.zeros(x[0].shape)
        y += p[0]*x[0]**0
        i = 1
        while i < order*nindeps+1:
            for o in range(1,order+1):
                for n in range(nindeps):
                    y+=p[i]*x[n]**o
                    i+=1
    elif isinstance(x,(list,np.ndarray)):
        order = len(p)-1
        y = np.zeros(x.shape)
        o = 0
        while o <= order:
            y += p[o]*x**o
            o += 1
    return y

def normweights(weights):
    """
    Normalize an array of weights by dividing them by their sum.
    
    weights:     an array of weights
    
    Returns an array of the normalized weights.
    """
    return weights/np.sum(weights)

def genresidual(weights,residuals):
    """
    Generate a single residual value from an array of fit residuals weighted
    according to an array of weights.
    
    weights:     an array of weights
    residuals:   an array of residuals from a fit for a single star 
    
    Returns a single residual value.
    """
    return np.sum(weights*residuals)

def magenresidual(weights,residuals):
    """
    Generate an array of residual values from a 2D array of fit residuals 
    weighted according to an array of weights.
    
    weights:     an array of weights
    residuals:   a 2D array of residuals from a fit, where the 0-axis spans 
                 pixel and the 1-axis spans stars
                 
    Returns an array of residual values
    """
    return np.array(np.matrix(weights)*np.matrix(residuals))[0]

def hist2d(x,y,nbins = 50 ,saveloc = '',labels=[]):
    """
    Creates a 2D histogram from data given by numpy's histogram

    x,y:        two 2D arrays to correlate
    nbins:      number of bins
    saveloc:    place to save histogram plot - if unspecified, do not
                save plot (kwarg, default = '')
    labels:     labels for histogram plot, with the following format
                [title,xlabel,ylabel,zlabel] - if unspecified, do 
                not label plot (kwarg, default = [])

    Returns the edges of the histogram bins and the 2D histogram

    """
    # Create histogram
    H,xedges,yedges = np.histogram2d(x,y,bins=nbins)
    # Reorient appropriately
    H = np.rot90(H)
    H = np.flipud(H)
    # Mask zero value bins
    Hmasked = np.ma.masked_where(H==0,H)
    # Begin creating figure
    plt.figure(figsize=(12,10))
    # Make histogram pixels with logscale
    plt.pcolormesh(xedges,yedges,Hmasked,
                   norm = LogNorm(vmin = Hmasked.min(),
                                  vmax = Hmasked.max()),
                   cmap = plt.get_cmap('Spectral_r'))
    # Create fit line x-array
    uplim = np.max(x)+5
    dolim = np.min(x)-5
    # Set plot limits
    plt.xlim(dolim+5,uplim-5)
    plt.ylim(np.min(y),np.max(y))
    # Add colourbar
    cbar = plt.colorbar()
    # Add labels
    if labels != []:
        title,xlabel,ylabel,zlabel = labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        cbar.ax.set_ylabel(zlabel)
    # Save plot
    if saveloc != '':
        plt.savefig(saveloc)
    plt.close()
    # Return histogram
    return xedges,yedges,Hmasked