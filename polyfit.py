import numpy as np

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

def idealerrs(p,x,err,order=1):
    X = makematrix(x)
    C = np.diag(err**2)
    return np.array(np.linalg.inv(X.T*C*X))

def bootstrap(p,x,y,err,order=1,ntrials = 10):
    params = np.zeros((ntrials,len(p)))
    for n in ntrials:
        sample = np.random.choice(x,size = x.shape,replace = True,p = None) #set p = 1/err**2?
        params[n] = regfit(sample,y,err = err,order = order)
    return (1./ntrials)*np.sum((params-p)**2,axis = 0)
    
def jackknife(p,x,y,err,order=1):
    params = np.zeros((len(x),len(p)))
    for n in range(len(x)):
        params[n] = regfit(np.delete(x,n),np.delete(y,n),err = np.delete(err,n),order=order)
    finalparam = (1./len(x))*np.sum(params,axis = 0)
    return finalparam,((len(x)-1)/len(x))*np.sum((params-finalparam)**2,axis = 0)


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
